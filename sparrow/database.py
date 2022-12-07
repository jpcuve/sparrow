from typing import List

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def read_test_values() -> List[str]:
    query = db.engine.execute('select text from test')
    return [r[0] for r in query]
