from flask.testing import FlaskClient

BASE_URL = 'http://localhost:5000/api/v1'


def test_list_instances(client: FlaskClient):
    res = client.get(f'{BASE_URL}/ec2-instances', headers={'x-api-key': 'vicky-api-key'})
    assert res.status_code // 100 == 2
    data = res.json
    print(data)

