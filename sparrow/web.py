from flask import Blueprint, jsonify

from sparrow import auth

bp = Blueprint('web', __name__, url_prefix='/web')


@bp.route('/')
@auth.login_required
def web_status():
    return jsonify(status='ok')


@bp.route('/perpetual')
@auth.login_required
def web_perpetual():
    return {}
