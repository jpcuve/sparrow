from flask import Blueprint, jsonify

bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')


@bp.route('/')
def api_index():
    return jsonify(status='ok')


@bp.route('/upload', methods=['POST'])
def api_upload():
    return jsonify(status='ok')
