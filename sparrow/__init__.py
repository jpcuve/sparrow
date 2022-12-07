import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

CONFIGURATION_LOCATION = 'FLASK_CONFIG'


def create_app() -> Flask:
    flask = Flask(__name__, static_folder='../app/build', static_url_path='/')
    flask.config.from_object('config')
    if CONFIGURATION_LOCATION in os.environ.keys():  # if FLASK_CONFIG defined in environment
        flask.config.from_envvar(CONFIGURATION_LOCATION)  # then overwrite flask.config from that file
    flask.development = os.environ.get('FLASK_ENV') == 'development'
    if flask.development:
        CORS(flask)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG if flask.development else logging.INFO)
    # from here on log using app.logger (or 'current_app.logger' in the context of a http request)
    flask.logger.info(f"Should we use the cloud? {flask.config['CLOUD']}")

    # db initilization (this is going to use the SQL_ALCHEMY_... variables from the configuration)
    # in production you use some other config.py file pointed to by the FLASK_CONFIG env variable
    from sparrow.database import db
    db.init_app(flask)

    # blueprint initializations (API endpoints, one file per version)
    from sparrow import api_v1
    flask.register_blueprint(api_v1.bp)

    # the following setup code is when we are in development mode only, with an in memory database
    # in that case, create tables and fill them with test data
    if flask.development:
        with flask.app_context():
            connection = db.engine.connect()
            with flask.open_resource('init.sql', 'rb') as resource:
                sql = str(resource.read(), 'utf8')
                for statement in sql.split(';'):
                    with connection.begin() as transaction:
                        try:
                            connection.execute(statement)
                            transaction.commit()
                        except Exception as e:
                            flask.logger.error(e)
                            transaction.rollback()

    @flask.errorhandler(Exception)
    def handle_error(ex):
        return jsonify(error=str(ex)), 500

    return flask
