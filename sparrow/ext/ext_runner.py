from flask import Flask

from sparrow.database import db_sparrow


class Runner:
    def __init__(self, app: Flask = None):
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        pass

    def submit(self, key: str, fn, *args, **kwargs):
        with db_sparrow.engine.connect() as conn:
            if db_sparrow.acquire_lock(conn, key):
                try:
                    fn(*args, **kwargs)
                finally:
                    db_sparrow.release_lock(conn, key)


runner = Runner()
