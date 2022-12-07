import uuid

from flask import Blueprint, jsonify, request

from sparrow.database import db_sparrow
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


@bp.route('/upload', methods=['POST'])
@user_feed
def api_upload(user_id: int):
    payload = request.json
    image_ids = []
    with db_sparrow.engine.connect() as conn:
        for data in payload:
            prompts = data.get('prompts', [])
            image = data.get('image')
            image_id = db_sparrow.insert_image(conn, user_id, prompts, image)
            image_ids.append(image_id)
    return jsonify(image_ids=image_ids)


@bp.route('/train', methods=['POST'])
@user_feed
def api_train(user_id: int):
    train_parameters = request.json
    with db_sparrow.engine.connect() as conn:
        train_id = db_sparrow.insert_training_request(conn, user_id, train_parameters)
    return jsonify(train_id=train_id)


@bp.route('/train-status/<train_id>')
@user_feed
def api_train_status(user_id: int, train_id: str):
    with db_sparrow.engine.connect() as conn:
        completed = db_sparrow.get_training_status(conn, user_id, train_id)
    return jsonify(completed=completed)


@bp.route('/infer/<train_id>', methods=['POST'])
@user_feed
def api_infer(user_id: int, train_id: str):
    payload = request.json
    prompt = payload['prompt']
    # here you must run your inference, that I will assume returns images
    # it is better though you return identifiers to images available on a CDN
    return jsonify(image_ids=[1, 2, 3])
