import pytest
from flask.testing import FlaskClient

from sparrow import create_app


@pytest.fixture(scope="module")
def client() -> FlaskClient:
    # create test app on port 5000
    app = create_app()
    with app.test_client() as client:
        yield client

# do not remove following commented lines as they require some thought

# def start_app(port: int):
#     app = create_app()
#     app.run(port=port)
#
#
# @pytest.fixture(scope="module")
# def sparrow_client() -> Client:
#     port = 5000
#     process = Process(target=start_app, args=(port,))
#     process.start()
#     # wait for server to be up
#     while True:
#         try:
#             requests.get(f'http://localhost:{port}/api/status')
#             break
#         except:
#             pass
#     # server is up
#     client = Client(f'localhost:{port}', 'vicky-api-key')
#     yield client
#     process.terminate()
