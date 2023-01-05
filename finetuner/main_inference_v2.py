#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import random
import matplotlib.pyplot as plt

from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler


# In[12]:


# Change these as you want:
model_path = "/home/ec2-user/finetuner/outputs/models/appyhigh_sample_person_2/2000"


# In[13]:


# Try our original prompt and make sure it works okay:
# prompt = "closeup photo of ggd woman in the garden on a bright sunny day"
# Setup the scheduler and pipeline
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False, set_alpha_to_one=False, steps_offset=1)
pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                               scheduler=scheduler,
                                               safety_checker=None,
                                               torch_dtype=torch.float16,
                                               revision="fp16").to("cuda")


# In[14]:


prompt = """symmetry! closeup portrait of sks man as a male wizard, high fantasy, intricate, elegant, highly detailed, photorealistic, cinematic lighting, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha,"""
negative_prompt = "ugly, tiling, airpods, earphone, cross eyed, squint, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft,cartoon, 3d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, blurry, boring, sketch, lackluster, repetitive, cropped"
num_samples = 6
guidance_scale = 13.5
num_inference_steps = 150
height = 512
width = 512

# Generate the images:
# Add height and width. Remove image and strength
images = pipe(prompt, height=height, width=width, negative_prompt=negative_prompt, num_images_per_prompt=num_samples,
              num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images


# In[15]:


num_rows, num_columns = 2, 3
fig_width, fig_height = 12, 6
fig, axarr = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))
axarr = axarr.flatten()
for i in range(num_rows*num_columns):
    _ = axarr[i].imshow(images[i])
    axarr[i].axis('off')


# In[ ]:



