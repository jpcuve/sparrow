from flask.testing import FlaskClient

BASE_URL = 'http://localhost:5000/api/v1'


def test_presence(client: FlaskClient):
    # getting status
    res = client.get(f'{BASE_URL}/')
    assert res.status_code // 100 == 2
    data = res.json
    print(data)


def test_protection(client: FlaskClient):
    res = client.get(f'{BASE_URL}/protected')
    assert res.status_code // 100 == 5
    data = res.json
    assert data.get('error') is not None
    res = client.get(f'{BASE_URL}/protected', headers={'x-api-key': 'anything'})
    assert res.status_code // 100 == 5
    data = res.json
    assert data.get('error') is not None
    res = client.get(f'{BASE_URL}/protected', headers={'x-api-key': 'jp-api-key'})
    assert res.status_code // 100 == 2

