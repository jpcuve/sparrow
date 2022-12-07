import pytest
from flask.testing import FlaskClient

from sparrow import create_app


@pytest.fixture(scope="module")
def client() -> FlaskClient:
    # create test app on port 5000
    app = create_app()
    with app.test_client() as client:
        yield client
