import functools
from typing import Callable

from flask import request


def user_feed(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        json_vectors = request.json
        return fn(None, *args, **kwargs)

    return wrapper
