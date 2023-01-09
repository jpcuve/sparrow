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
    return jsonify(status='ok')


@bp.route('/ec2-instances')
def api_ec2_instances():
    return ec2.find_instances()


@bp.route('/ec2-instance/<instance_id>/<verb>')
def api_ec2_instance(instance_id: str, verb: str):
    if verb == 'start':
        ec2.start_instance(instance_id)
    elif verb == 'stop':
        ec2.stop_instance(instance_id)
    else:
        raise RuntimeError(f"Unknown command: {verb}")
    return jsonify(status='ok')