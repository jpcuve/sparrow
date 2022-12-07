import functools
from typing import Callable

from flask import request

from sparrow.database import db_sparrow


def user_feed(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # get api key from header
        api_key = request.headers.get('x-api-key')
        if api_key is None:
            raise RuntimeError("Missing api key")
        # find user based on api key
        with db_sparrow.engine.connect() as conn:
            user_id = db_sparrow.get_user_id(conn, api_key)
        if user_id is None:
            raise RuntimeError("User not found")
        return fn(user_id, *args, **kwargs)

    return wrapper
