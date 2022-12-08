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
        # read prompts from .txt file with same name, if exists
        prompts = []
        path_txt = path / '..' / path.parts[-1].replace('.png', '.txt')
        if path_txt.exists():
            with open(path_txt, 'r') as f:
                for line in f.readlines():
                    prompts.append(line.strip())
        payload.append({
            'prompts': prompts,
            'image': base64.b64encode(data).decode('ascii')
        })
    res = client.post(f'{BASE_URL}/upload', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    assert len(data.get('image_ids', [])) == 3


def test_training_request(client: FlaskClient):
    payload = {
        'parameter1': 'value1',
        'parameter2': 'value2',
    }
    res = client.post(f'{BASE_URL}/train', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    assert len(data.get('train_id', '')) > 0


def get_training_id(client: FlaskClient) -> str:
    payload = {
        'parameter1': 'value1',
        'parameter2': 'value2',
    }
    res = client.post(f'{BASE_URL}/train', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    train_id = res.json.get('train_id')
    assert train_id is not None
    return train_id


def test_training_status(client: FlaskClient):
    train_id = get_training_id(client)
    res = client.get(f'{BASE_URL}/train-status/{train_id}', headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    assert data.get('completed', -1) == 0


def test_infer(client: FlaskClient):
    train_id = get_training_id(client)
    payload = {
        'prompt': 'astronaut riding a horse on Mars'
    }
    res = client.post(f'{BASE_URL}/infer/{train_id}', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    assert data.get('image_ids') is not None
