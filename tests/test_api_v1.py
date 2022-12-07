from flask.testing import FlaskClient

BASE_URL = 'http://localhost:5000/api/v1'


def test_presence(client: FlaskClient):
    # getting status
    res = client.get(f'{BASE_URL}/')
    assert res.status_code // 100 == 2
    data = res.json
    print(data)
