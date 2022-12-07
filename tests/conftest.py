from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import pytest
import requests
from flask.testing import FlaskClient

from sparrow import create_app


def start_app(port: int):
    app = create_app()
    app.run(port=port)


@pytest.fixture(scope="module")
def client() -> FlaskClient:
    processes = []
    port_range = range(5001, 5006)
    for port in port_range:
        processes.append(Process(target=start_app, args=(port,)))
    for p in processes:
        p.start()
    # wait for flask apps to be up
    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        url_list = [f'http://localhost:{port}/api/status' for port in port_range]
        executor.map(lambda url: requests.get(url), url_list)
    # create test app on port 5000
    app = create_app()
    with app.test_client() as client:
        yield client
    for p in processes:
        p.terminate()
