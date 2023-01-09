from flask import Blueprint, jsonify, request

from sparrow.database import db_sparrow
from sparrow.ext.ext_ec2 import ec2
from sparrow.helper import user_feed

bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')


@bp.route('/')
def api_index():
    return jsonify(status='ok')


@bp.route('/protected')
@user_feed
def api_protected(user_id: int):
    print(f"User id: {user_id}")
    return jsonify(status='ok')


@bp.route('/finetune-job', methods=['POST'])
@user_feed
def api_finetune_job(user_id: int):
    payload = request.json
    model_reference = payload.get('model_reference')
    gender = payload.get('gender', '')
    max_train_steps = payload.get('max_train_steps', 5000)
    image_urls = payload.get('image_urls', [])
    if model_reference is None or len(image_urls) == 0:
        raise RuntimeError('Missing parameter')
    with db_sparrow.engine.connect() as conn:
        finetune_job_id = db_sparrow.insert_finetune_job(
            conn, user_id, model_reference, gender, max_train_steps, image_urls)
        db_sparrow.insert_finetune_job_event(conn, finetune_job_id, 'SUBMITTED')
    return jsonify(finetune_job_id=finetune_job_id)


@bp.route('/finetune-job-status/<finetune_job_id>')
@user_feed
def api_finetune_job_status(user_id: int, finetune_job_id: int):
    with db_sparrow.engine.connect() as conn:
        status = db_sparrow.find_finetune_job_status(conn, user_id, finetune_job_id)
    return jsonify(status=status)


@bp.route('/inference-job', methods=['POST'])
@user_feed
def api_inference_job(user_id: int):
    payload = request.json
    model_reference = payload.get('model_reference')
    prompt = payload.get('prompt')
    negative_prompt = payload.get('negative_prompt')
    num_inference_steps = payload.get('num_inference_steps', 150)
    num_images_per_prompt = payload.get('num_images_per_prompt', 6)
    guidance_scale = payload.get('guidance_scale', 6.5)
    if model_reference is None or prompt is None or negative_prompt is None:
        raise RuntimeError('Missing parameter')
    with db_sparrow.engine.connect() as conn:
        inference_job_id = db_sparrow.insert_inference_job(
            conn, user_id, model_reference, prompt, negative_prompt, num_inference_steps, num_images_per_prompt,
            guidance_scale)
        db_sparrow.insert_inference_job_event(conn, inference_job_id, 'SUBMITTED')
    return jsonify(inference_job_id=inference_job_id)


@bp.route('/inference-job-status/<inference_job_id>')
@user_feed
def api_inference_job_status(user_id: int, inference_job_id: int):
    with db_sparrow.engine.connect() as conn:
        status = db_sparrow.find_inference_job_status(conn, user_id, inference_job_id)
    return jsonify(status=status)


@bp.route('/generated-images/<inference_job_id>')
@user_feed
def api_generated_images(user_id: int, inference_job_id: int):
    with db_sparrow.engine.connect() as conn:
        image_urls = db_sparrow.find_generated_image_urls(conn, user_id, inference_job_id)
    return jsonify(image_urls=image_urls)
