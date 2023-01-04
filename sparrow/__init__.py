import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash

CONFIGURATION_LOCATION = 'FLASK_CONFIG'

db = SQLAlchemy()

auth = HTTPBasicAuth()

def create_app() -> Flask:
    app = Flask(__name__, static_folder='../app/build', static_url_path='/')
    app.config.from_object('config')
    if CONFIGURATION_LOCATION in os.environ.keys():  # if FLASK_CONFIG defined in environment
        app.config.from_envvar(CONFIGURATION_LOCATION)  # then overwrite flask.config from that file
    app.development = os.environ.get('FLASK_ENV') == 'development'
    if app.development:
        CORS(app)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG if app.development else logging.INFO)
    # from here on log using app.logger (or 'current_app.logger' in the context of a http request)
    app.logger.info(f"Should we use the cloud? {app.config['CLOUD']}")

    # db initilization (this is going to use the SQL_ALCHEMY_... variables from the configuration)
    # in production you use some other config.py file pointed to by the FLASK_CONFIG env variable
    db.init_app(app)
    from sparrow.database import db_sparrow
    db_sparrow.init_app(app)

    # web application initialization
    from sparrow import web
    app.register_blueprint(web.bp)

    # blueprint initializations (API endpoints, one file per version)
    from sparrow import api_v1
    app.register_blueprint(api_v1.bp)

    # the following setup code is when we are in development mode only, with an in memory database
    # in that case, create tables and fill them with test data
    if app.development:
        db_sparrow.create_all()
        with db_sparrow.engine.connect() as connection:
            res = next(connection.execute('select count(*) from users'))
            if res[0] == 0:
                with app.open_resource('data.sql', 'rb') as resource:
                    sql = str(resource.read(), 'utf8')
                    for statement in sql.split(';'):
                        with connection.begin() as transaction:
                            try:
                                connection.execute(statement)
                                transaction.commit()
                            except Exception as e:
                                app.logger.error(e)
                                transaction.rollback()

    @auth.verify_password
    def verify_password(username: str, password: str) -> str:
        users = app.config['WEB_USERS']
        if username in users and check_password_hash(users[username], password):
            return username

    @app.errorhandler(Exception)
    def handle_error(ex):
        return jsonify(error=str(ex)), 500

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.errorhandler(404)
    def not_found(e):
        return app.send_static_file('index.html')

    return app
