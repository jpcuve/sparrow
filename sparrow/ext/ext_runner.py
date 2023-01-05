from concurrent.futures import ThreadPoolExecutor

from flask import Flask

from sparrow.database import db_sparrow


def run(key: str, fn, *args, **kwargs):
    with db_sparrow.engine.connect() as conn:
        if db_sparrow.acquire_lock(conn, key):
            try:
                fn(*args, **kwargs)
            finally:
                db_sparrow.release_lock(conn, key)


class Runner:
    def __init__(self, app: Flask = None):
        self.executor = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        self.executor = ThreadPoolExecutor(max_workers=4)

    def submit(self, key: str, fn, *args, **kwargs):
        self.executor.submit(run, key, fn, *args, **kwargs)


runner = Runner()
