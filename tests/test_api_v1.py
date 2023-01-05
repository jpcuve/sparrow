import uuid

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


def test_finetune_request(client: FlaskClient):
    payload = {
        'model_reference': uuid.uuid4(),
        'image_urls': ['url-1', 'url-2', 'url-3'],
    }
    res = client.post(f'{BASE_URL}/finetune-job', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    finetune_job_id = data.get('finetune_job_id')
    assert finetune_job_id is not None
    res = client.get(f'{BASE_URL}/finetune-job-status/{finetune_job_id}', headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    status = data.get('status')
    assert status == 'SUBMITTED'


def test_inference_request(client: FlaskClient):
    model_reference = uuid.uuid4()
    payload = {
        'model_reference': model_reference,
        'image_urls': ['url-4', 'url-5', 'url-6'],
    }
    res = client.post(f'{BASE_URL}/finetune-job', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    finetune_job_id = data.get('finetune_job_id')
    assert finetune_job_id is not None
    payload = {
        'model_reference': model_reference,
        'prompt': 'My positive prompt',
        'negative_prompt': 'My negative prompt'
    }
    res = client.post(f'{BASE_URL}/inference-job', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    inference_job_id = data.get('inference_job_id')
    assert inference_job_id is not None
    res = client.get(f'{BASE_URL}/inference-job-status/{inference_job_id}', headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    status = data.get('status')
    assert status == 'SUBMITTED'


def test_generated_image_urls(client: FlaskClient):
    model_reference = uuid.uuid4()
    payload = {
        'model_reference': model_reference,
        'image_urls': ['url-4', 'url-5', 'url-6'],
    }
    res = client.post(f'{BASE_URL}/finetune-job', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    finetune_job_id = data.get('finetune_job_id')
    assert finetune_job_id is not None
    payload = {
        'model_reference': model_reference,
        'prompt': 'My positive prompt',
        'negative_prompt': 'My negative prompt'
    }
    res = client.post(f'{BASE_URL}/inference-job', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    inference_job_id = data.get('inference_job_id')
    assert inference_job_id is not None
    res = client.get(f'{BASE_URL}/generated-images/{inference_job_id}', headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    print(data)
