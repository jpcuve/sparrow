import uuid

from flask import Blueprint, jsonify, request

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
    return jsonify(status='ok')


@bp.route('/train', methods=['POST'])
@user_feed
def api_train(user_id: int):
    train_id = str(uuid.uuid4())
    train_parameters = request.json

    return jsonify(id=train_id)


@bp.route('/train-status/<train_id>')
@user_feed
def api_train_status(user_id: int, train_id: str):
    return jsonify(completed=100)


@bp.route('/infer/<train_id>')
@user_feed
def api_infer(user_id: int, train_id: str):
    return jsonify(status='ok')
