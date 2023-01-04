from pathlib import Path

from werkzeug.security import generate_password_hash

SQLALCHEMY_BINDS = {
    'sparrow': 'sqlite:///../sparrow.sqlite'
}
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True
CLOUD = False
WEB_USERS = {
    'jpc@tilleuls': generate_password_hash('jp33ere'),
    'vicky': generate_password_hash('another_password'),
}
