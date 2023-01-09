from flask import Blueprint, request, jsonify

from sparrow.database import db_sparrow
from sparrow.ext.ext_ec2 import ec2

bp = Blueprint('api_internal', __name__, url_prefix='/api/internal')


@bp.route('/update-finetune-job-status/<finetune_job_id>', methods=['POST'])
def api_update_finetune_job_status(finetune_job_id: str):
    payload = request.json
    status = payload.get('status')
    progress = payload.get('progress')
    comment = payload.get('comment')
    if status is not None and progress is not None:
        with db_sparrow.engine.connect() as conn:
            db_sparrow.insert_finetune_job_event(conn, finetune_job_id, status, progress, comment)
            if progress >= 1.0:
                finetune_jobs = db_sparrow.find_finetune_jobs(conn, id=finetune_job_id)
                if len(finetune_jobs) > 0 and finetune_jobs[0]['aws_instance_id'] is not None:
                    db_sparrow.reference_aws_instance(finetune_jobs[0]['aws_instance_id'], None)
    return jsonify(status='ok')


@bp.route('/update-inference-job-status/<inference_job_id>', methods=['POST'])
def api_update_inference_job_status(inference_job_id: str):
    payload = request.json
    status = payload.get('status')
    progress = payload.get('progress')
    comment = payload.get('comment')
    if status is not None and progress is not None:
        with db_sparrow.engine.connect() as conn:
            db_sparrow.insert_inference_job_event(conn, inference_job_id, status, progress, comment)
            if progress >= 1.0:
                inference_jobs = db_sparrow.find_inference_jobs(conn, id=inference_job_id)
                if len(inference_jobs) > 0 and inference_jobs[0]['aws_instance_id'] is not None:
                    db_sparrow.reference_aws_instance(inference_jobs[0]['aws_instance_id'], None)
    return jsonify(status='ok')


@bp.route('/ec2-instances')
def api_ec2_instances():
    return ec2.find_instances()
