from flask import Blueprint, jsonify

from sparrow import auth, long_running_task
from sparrow.ext.ext_runner import runner

bp = Blueprint('web', __name__, url_prefix='/web')


@bp.route('/')
@auth.login_required
def web_status():
    return jsonify(status='ok')


@bp.route('/perpetual')
@auth.login_required
def web_perpetual():
    return {}


@bp.route('/long-process')
def web_long_process():
    runner.submit('test', long_running_task, 10)
    return jsonify(status='ok')
