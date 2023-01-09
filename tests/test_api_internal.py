import uuid

from flask.testing import FlaskClient

CLIENT_BASE_URL = 'http://localhost:5000/api/v1'
BASE_URL = 'http://localhost:5000/api/internal'


def test_list_instances(client: FlaskClient):
    res = client.get(f'{BASE_URL}/ec2-instances')
    assert res.status_code // 100 == 2
    data = res.json
    print(data)


def test_status(client: FlaskClient):
    # create a finetune job
    payload = {
        'model_reference': uuid.uuid4(),
        'image_urls': ['url-1', 'url-2', 'url-3'],
    }
    res = client.post(f'{CLIENT_BASE_URL}/finetune-job', json=payload, headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    finetune_job_id = data.get('finetune_job_id')
    assert finetune_job_id is not None
    for progress in range(10):
        status = f'P-{progress}'
        res = client.post(f'{BASE_URL}/update-finetune-job-status/{finetune_job_id}', json={
            'status': status,
            'progress': progress / 10.0,
            'comment': f'Step n°{progress + 1}'
        })
        assert res.status_code // 100 == 2
        res = client.get(f'{CLIENT_BASE_URL}/finetune-job-status/{finetune_job_id}',
                         headers={'x-api-key': 'vicky-api-key'})
        assert res.status_code // 100 == 2
        data = res.json
        read_status = data.get('status')
        assert status == read_status
