from pathlib import Path

SQLALCHEMY_BINDS = {
    'sparrow': 'sqlite:///../sparrow.sqlite'
}
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True
GOOGLE_APPLICATION_CREDENTIALS = str(Path.home() / 'messio-d8d865cf4d63.json')
CLOUD = False
