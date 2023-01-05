#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Things to check
# 1. Whether the algorithm uses regularization images from the given folder or whether it samples new images
# 2. Check whether the commandline arguments match with that of our config
# 3. Check if gradient checkpointing is needed or not
# 4. vae that we load is from pretrained_model_name_or_path whereas we compute latents from pretrained_vae_name_or_path. We need to look into why this discrepancy
# 5. UNet is loaded in torch.float32 whereas everything else is in fp16
# 6. Checkout the table on performance metrics and choose the right parameters to train
# 7. Experiment tracking reference: https://huggingface.co/docs/accelerate/usage_guides/tracking
# 8. Verify is the batch size one or two when the train_batch_size is 1 and with_prior_preservation is True


# In[ ]:


# Changes from v1 and v2
# 'pretrained_model_name_or_path': 'stabilityai/stable-diffusion-2-base'
# 'pretrained_vae_name_or_path': 'stabilityai/sd-vae-ft-mse'
# 'pad_tokens': False
# vae encoder runs with torch.no_grad()
# DDPM scheduler vs DDIM scheduler
# Try out the class conditioned prompt and see what kind of images does the model generate
# DDPM scheduler is used in training and DDIM scheduler is used in inference


# In[ ]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--gender')
parser.add_argument('--max_train_steps')
args = parser.parse_args()

name = args.name
gender = args.gender
max_train_steps = int(args.max_train_steps)
print(f'Processing: name: {name}, gender: {gender}, max_train_steps: {max_train_steps}')

# In[ ]:


config = {'pretrained_model_name_or_path': 'stabilityai/stable-diffusion-2-base',
          'pretrained_vae_name_or_path': 'stabilityai/sd-vae-ft-mse',
          'tokenizer_name': None,
          'concepts_list': [
              {
                  'instance_prompt': f'photo of sks {gender}',
                  'class_prompt': f'photo of a {gender}',
                  'instance_data_dir': f"/home/ec2-user/data/stylize-private/preprocessed_inputs/{name}/finalized_images/",
                  'class_data_dir': f"/home/ec2-user/finetuner/data/regularization_images/{gender}"
                  # Path of regularization images
              }
          ],
          'output_dir': f'/home/ec2-user/finetuner/outputs/models/{name}',
          'logging_dir': '/home/ec2-user/finetuner/logs',
          'output_images_dir': f'/home/ec2-user/finetuner/outputs/images/{name}',
          'save_sample_prompt': f'closeup portrait of sks {gender} in an ocean',
          'save_sample_negative_prompt': 'ugly, tiling, hands, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, multiple people, watermark, grainy, signature, cut off, draft',
          'n_save_sample': 4,
          'save_guidance_scale': 8.5,
          'save_infer_steps': 150,
          'pad_tokens': False,
          'with_prior_preservation': True,
          'prior_loss_weight': 1.0,
          'num_class_images': 200,
          'seed': None,
          'resolution': 512,
          'center_crop': False,
          'train_batch_size': 1,
          'sample_batch_size': 4,
          'num_train_epochs': None,
          'max_train_steps': int(max_train_steps),
          'gradient_accumulation_steps': 1,
          'gradient_checkpointing': False,
          'learning_rate': 2e-6,
          'scale_lr': False,
          'lr_scheduler_type': 'cosine',
          'lr_warmup_steps': 0,
          'use_8bit_adam': True,
          'adam_beta1': 0.9,
          'adam_beta2': 0.999,
          'adam_weight_decay': 1e-2,
          'adam_epsilon': 1e-08,
          'max_grad_norm': 1.0,
          'log_interval': 10,
          'save_interval': 200,
          'save_min_steps': 999,  # Start saving weights after save_min_steps
          'mixed_precision': 'fp16',
          'revision': None,
          'hflip': True,
          'train_text_encoder': True,
          'not_cache_latents': False,
          }

# In[ ]:


import math
import torch
import hashlib
import itertools
import bitsandbytes as bnb
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from accelerate import Accelerator
from contextlib import nullcontext
from torch.utils.data import Dataset
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from dreambooth_dataset_v2 import AverageMeter
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from dreambooth_dataset_v2 import LatentsDataset, DreamBoothDataset, PromptDataset
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

# In[ ]:


# Step 1: Initialize Accelerator
accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'],
                          mixed_precision=config['mixed_precision'],
                          log_with="wandb",
                          logging_dir=config['logging_dir'])

# In[ ]:


# Step 2: Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
if config['train_text_encoder'] and config['gradient_accumulation_steps'] > 1 and accelerator.num_processes > 1:
    raise ValueError(
        "Gradient accumulation is not supported when training the text encoder in distributed training. "
        "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
    )

# In[ ]:


# Step 3: Set seed
if config['seed'] is not None:
    set_seed(config['seed'])

# In[ ]:


# Step 4: Sample additional images for prior preservation
if config['with_prior_preservation']:
    pipeline = None
    for concept in config['concepts_list']:
        class_images_dir = Path(concept['class_data_dir'])
        class_images_dir.mkdir(parents=True, exist_ok=True)
        num_current_class_images = len(list(Path(concept['class_data_dir']).iterdir()))
        print(f"Minimum number of class images required: {config['num_class_images']}")
        print(f"Number of images in the current class concept: {num_current_class_images}")
        if num_current_class_images < config['num_class_images']:
            print(f"Sampling the remaining {config['num_class_images'] - num_current_class_images}")
            torch_dtype = torch.float16 if accelerator.device.type == 'cuda' else torch.float32
            if pipeline is None:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    config['pretrained_model_name_or_path'],
                    vae=AutoencoderKL.from_pretrained(
                        config['pretrained_vae_name_or_path'] or config['pretrained_model_name_or_path'],
                        subfolder=None if config['pretrained_vae_name_or_path'] else 'vae',
                        revision=None if config['pretrained_vae_name_or_path'] else config['revision'],
                        torch_dtype=torch_dtype
                    ),
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=config['revision']
                )
                pipeline.set_progress_bar_config(disable=True)
                pipeline.to(accelerator.device)
                sample_dataset = PromptDataset(concept['class_prompt'],
                                               config['num_class_images'] - num_current_class_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=config['sample_batch_size'])
                sample_dataloader = accelerator.prepare(sample_dataloader)
                with torch.autocast('cuda'), torch.inference_mode():
                    for example in tqdm(sample_dataloader, desc='Generating class images',
                                        disable=not accelerator.is_local_main_process):
                        images = pipeline(example['prompt']).images
                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filepath = Path(concept['class_data_dir']) / f'{hash_image}.jpg'
                            image.save(image_filepath)
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# In[ ]:


# Step 5: Load the tokenizer
if config['tokenizer_name']:
    tokenizer = CLIPTokenizer.from_pretrained(config['tokenizer_name'],
                                              revision=config['revision'])
elif config['pretrained_model_name_or_path']:
    tokenizer = CLIPTokenizer.from_pretrained(config['pretrained_model_name_or_path'],
                                              subfolder='tokenizer',
                                              revision=config['revision'])

# In[ ]:


# Step 6: Load models required for constructing Stable Diffusion
# Models: text_encoder, vae, unet
text_encoder = CLIPTextModel.from_pretrained(config['pretrained_model_name_or_path'],
                                             subfolder='text_encoder',
                                             revision=config['revision'])
vae = AutoencoderKL.from_pretrained(config['pretrained_model_name_or_path'],
                                    subfolder='vae',
                                    revision=config['revision'])
unet = UNet2DConditionModel.from_pretrained(config['pretrained_model_name_or_path'],
                                            subfolder='unet',
                                            revision=config['revision'],
                                            torch_dtype=torch.float32)
noise_scheduler = DDPMScheduler.from_pretrained(config['pretrained_model_name_or_path'],
                                                subfolder='scheduler')

# In[ ]:


# Step 7: Enable Gradient checkpointing
if config['gradient_checkpointing']:
    unet.enable_gradient_checkpointing()
    if config['train_text_encoder']:
        text_encoder.enable_gradient_checkpointing()

# In[ ]:


# Step 8: Scaling Learning rate
if config['scale_lr']:
    config['learning_rate'] = (config['learning_rate'] * config['gradient_accumulation_steps'] * config[
        'train_batch_size'] * accelerator.num_processes)

# In[ ]:


# Step 9: Using 8 Bit Adam Optimizer
if config['use_8bit_adam']:
    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# In[ ]:


# Step 10: Choose the parameters to optimize and define the optimizer
vae.requires_grad_(False)
if not config['train_text_encoder']:
    text_encoder.requires_grad_(False)

params_to_optimize = (
    itertools.chain(unet.parameters(), text_encoder.parameters()) if config['train_text_encoder'] else unet.parameters()
)

optimizer = optimizer_class(params_to_optimize,
                            lr=config['learning_rate'],
                            betas=(config['adam_beta1'], config['adam_beta2']),
                            weight_decay=config['adam_weight_decay'],
                            eps=config['adam_epsilon'])

# In[ ]:


# Step 11a: Define Dataset
train_dataset = DreamBoothDataset(concepts_list=config['concepts_list'],
                                  tokenizer=tokenizer,
                                  with_prior_preservation=config['with_prior_preservation'],
                                  size=config['resolution'],
                                  pad_tokens=config['pad_tokens'],
                                  center_crop=config['center_crop'],
                                  hflip=config['hflip'])


# In[ ]:


# Step 11b: Define DataLoader and collate_fn
def collate_fn(examples):
    input_ids = [example['instance_prompt_ids'] for example in examples]
    pixel_values = [example['instance_images'] for example in examples]
    if config['with_prior_preservation']:
        input_ids += [example['class_prompt_ids'] for example in examples]
        pixel_values += [example['class_images'] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = tokenizer.pad({'input_ids': input_ids}, padding=True, return_tensors='pt').input_ids
    batch = {'pixel_values': pixel_values, 'input_ids': input_ids}
    return batch


train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['train_batch_size'],
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               pin_memory=True)

# In[ ]:


# Step 12: Move the components which wouldn't be trained to lower precision
weight_dtype = torch.float32
if config['mixed_precision'] == "fp16":
    weight_dtype = torch.float16
elif config['mixed_precision'] == "bf16":
    weight_dtype = torch.bfloat16

vae.to(accelerator.device, dtype=weight_dtype)
if not config['train_text_encoder']:
    text_encoder.to(accelerator.device, dtype=weight_dtype)
else:
    text_encoder.to(accelerator.device)

# In[ ]:


# Step 13: Cache latents
if not config['not_cache_latents']:
    latents_cache, text_encoder_cache = [], []
    for batch in tqdm(train_dataloader, desc='Caching latents'):
        with torch.no_grad():
            batch['pixel_values'] = batch['pixel_values'].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
            batch['input_ids'] = batch['input_ids'].to(accelerator.device, non_blocking=True)
            latents_cache.append(vae.encode(batch['pixel_values']).latent_dist)
            if config['train_text_encoder']:
                text_encoder_cache.append(batch['input_ids'])
            else:
                text_encoder_cache.append(text_encoder(batch['input_ids']).last_hidden_state)
    train_dataset = LatentsDataset(latents_cache=latents_cache, text_encoder_cache=text_encoder_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train_batch_size'],
                                                   collate_fn=lambda x: x, shuffle=True)
    del vae
    if not config['train_text_encoder']:
        del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# In[ ]:


# Step 14: Calculate the number of training steps
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['gradient_accumulation_steps'])
if config['max_train_steps'] is None:
    config['max_train_steps'] = config['num_train_epochs'] * num_update_steps_per_epoch
    overrode_max_train_steps = True

# In[ ]:


# Step 15: Define LR scheduler
lr_scheduler = get_scheduler(config['lr_scheduler_type'],
                             optimizer=optimizer,
                             num_warmup_steps=config['lr_warmup_steps'] * config['gradient_accumulation_steps'],
                             num_training_steps=config['max_train_steps'] * config['gradient_accumulation_steps'])

# In[ ]:


# Step 16: Prepare Accelerator
if config['train_text_encoder']:
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )
else:
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

# In[ ]:


# Step 17: Recalculate the total training steps as the size of the training dataloader might have changed
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['gradient_accumulation_steps'])
if overrode_max_train_steps:
    config['max_train_steps'] = config['num_train_epochs'] * num_update_steps_per_epoch
config['num_train_epochs'] = math.ceil(config['max_train_steps'] / num_update_steps_per_epoch)


# In[ ]:


def save_weights(step):
    if accelerator.is_main_process:
        if config['train_text_encoder']:
            text_encoder_model = accelerator.unwrap_model(text_encoder)
        else:
            text_encoder_model = CLIPTextModel.from_pretrained(config['pretrained_model_name_or_path'],
                                                               subfolder="text_encoder",
                                                               revision=config['revision'])
        # Use DDPM Scheduler for training and DDIM scheduler for inference
        inference_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear',
                                            clip_sample=False, set_alpha_to_one=False)

        pipeline = StableDiffusionPipeline.from_pretrained(
            config['pretrained_model_name_or_path'],
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder_model,
            vae=AutoencoderKL.from_pretrained(
                config['pretrained_vae_name_or_path'] or config['pretrained_model_name_or_path'],
                subfolder=None if config['pretrained_vae_name_or_path'] else 'vae',
                revision=None if config['pretrained_vae_name_or_path'] else config['revision']
            ),
            safety_checker=None,
            scheduler=inference_scheduler,
            torch_dtype=torch.float16,
            revision=config['revision']
        )
        output_model_step_dir = Path(f"{config['output_dir']}/{step}")
        output_model_step_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save_pretrained(Path(config['output_dir']) / f'{step}')

        if config['save_sample_prompt'] is not None:
            output_images_step_dir = Path(f"{config['output_images_dir']}/{step}")
            output_images_step_dir.mkdir(parents=True, exist_ok=True)
            pipeline = pipeline.to(accelerator.device)
            g_cuda = torch.Generator(device=accelerator.device).manual_seed(
                config['seed'] if config['seed'] else torch.randint(0, 1024, (1, 1)).item())
            pipeline.set_progress_bar_config(disable=True)
            with torch.autocast("cuda"), torch.inference_mode():
                if config['save_sample_negative_prompt']:
                    images = pipeline(config['save_sample_prompt'],
                                      negative_prompt=config['save_sample_negative_prompt'],
                                      guidance_scale=config['save_guidance_scale'],
                                      num_inference_steps=config['save_infer_steps'],
                                      num_images_per_prompt=config['n_save_sample'],
                                      generator=g_cuda).images
                else:
                    images = pipeline(config['save_sample_prompt'],
                                      guidance_scale=config['save_guidance_scale'],
                                      num_inference_steps=config['save_infer_steps'],
                                      num_images_per_prompt=config['n_save_sample'],
                                      generator=g_cuda).images
                for i, image in enumerate(images):
                    image.save(output_images_step_dir / f'{i}.png')
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Weights saved at: {output_model_step_dir}")
        print(f"Images saved at: {output_images_step_dir}")


# In[ ]:


if accelerator.is_main_process:
    accelerator.init_trackers("dreambooth", config=config)

# In[ ]:


total_batch_size = config['train_batch_size'] * accelerator.num_processes * config['gradient_accumulation_steps']

# In[ ]:


print('****** Running training ******')
print(f'Number of training examples: {len(train_dataset)}')
print(f'Number of batches each epoch: {len(train_dataloader)}')
print(f"Number of epochs: {config['num_train_epochs']}")
print(f"Instantanaeous batch size per device: {config['train_batch_size']}")
print(f'Total train batch size including parallel, distribution and accumulation: {total_batch_size}')
print(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
print(f"Total optimization steps: {config['max_train_steps']}")

# In[ ]:


# Step 18: Setup training processes
progress_bar = tqdm(range(config['max_train_steps']), disable=not accelerator.is_local_main_process)
progress_bar.set_description('Steps')
global_step = 0
loss_avg = AverageMeter()
text_encoder_context = nullcontext() if config['train_text_encoder'] else torch.no_grad()

for epoch in range(config['num_train_epochs']):
    unet.train()
    if config['train_text_encoder']:
        text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Step 19a: Convert images to latent space
            with torch.no_grad():
                if not config['not_cache_latents']:
                    latent_dist = batch[0][0]
                else:
                    latent_dist = vae.encode(batch['pixel_values'].to(dtype=weight_dtype)).latent_dist
                latents = latent_dist.sample() * 0.18215
            # Step 19b: Sample noise that we will add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Step 19c: Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            # Step 19d: Forward diffusion process: Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Step 19e: Get the text embedding for conditioning
            with text_encoder_context:
                if not config['not_cache_latents']:
                    if config['train_text_encoder']:
                        encoder_hidden_states = text_encoder(batch[0][1]).last_hidden_state
                    else:
                        encoder_hidden_states = batch[0][1]
                else:
                    encoder_hidden_states = text_encoder(batch['input_ids']).last_hidden_state
            # Step 19f: Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # Step 19g: Compute the loss
            if config['with_prior_preservation']:
                # input noise_pred: torch.Size([2, 4, 64, 64])
                # output noise_pred: torch.Size([1, 4, 64, 64])
                noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                noise, noise_prior = torch.chunk(noise, 2, dim=0)
                # Compute instance loss
                instance_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()
                # Compute prior loss
                prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")
                # Compute weighted average of instance_loss and prior_loss
                loss = instance_loss + config['prior_loss_weight'] * prior_loss
            else:
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                # Step 19h: Backprogate loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_avg.update(loss.detach_(), bsz)
        logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        ####
        if (global_step % config['save_interval']) == 0 and (global_step > 0):
            save_weights(global_step)
            unet = accelerator.prepare(unet)
        ####

        progress_bar.update(1)
        global_step += 1
        if global_step >= config['max_train_steps']:
            break

        accelerator.wait_for_everyone()

# In[ ]:


####
save_weights(global_step)
####
accelerator.end_training()

# In[ ]:



