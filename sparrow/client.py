import requests


class Client:
    def __init__(self, host: str, api_key: str):
        self.host = host
        self.session = requests.Session()
        self.session.headers.update({'x-api-key': api_key})
        self.base_url = f'http://{host}/api/v1'  # decide the version

    def check(self) -> bool:
        res = self.session.get(self.base_url)
        return res.status_code // 100 == 2

