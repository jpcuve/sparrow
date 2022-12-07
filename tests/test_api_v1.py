import base64
from pathlib import Path

from flask.testing import FlaskClient

BASE_URL = 'http://localhost:5000/api/v1'


def test_presence(client: FlaskClient):
    # getting status
    res = client.get(f'{BASE_URL}/')
    assert res.status_code // 100 == 2
    data = res.json
    print(data)


def test_protection(client: FlaskClient):
    res = client.get(f'{BASE_URL}/protected')  # missing x-api-key header
    assert res.status_code // 100 == 5
    data = res.json
    assert data.get('error') is not None
    res = client.get(f'{BASE_URL}/protected', headers={'x-api-key': 'anything'})  # wrong x-api-key header
    assert res.status_code // 100 == 5
    data = res.json
    assert data.get('error') is not None
    res = client.get(f'{BASE_URL}/protected', headers={'x-api-key': 'jp-api-key'})  # correct
    assert res.status_code // 100 == 2


def test_image_upload(client: FlaskClient):
    payload = []
    for path in (Path('.') / 'images').glob('*.png'):
        with open(path, 'rb') as f:
            data = f.read()
        payload.append({
            'prompts': ['blue sparrow', 'blue bird'],
            'image': base64.b64encode(data).decode('ascii')
        })
    res = client.post(f'{BASE_URL}/upload', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    assert len(data.get('image_ids', [])) == 3
