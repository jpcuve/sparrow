import base64
from pathlib import Path

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

    def upload(self, folder_pathname: str):
        folder_path = Path(folder_pathname)
        if folder_path.exists():
            payload = []
            for extension in ['.png', '.jpg', '.jpeg']:
                for path in folder_path.glob(f'*{extension}'):
                    with open(path, 'rb') as f:
                        image_data = f.read()
                    path_txt = path.parent / path.parts[-1].replace(extension, '.txt')
                    prompts = []
                    if path_txt.exists():
                        with open(path_txt, 'r') as f:
                            for line in f.readlines():
                                prompts.append(line.strip())
                    payload.append({
                        'prompts': prompts,
                        'image': base64.b64encode(image_data).decode('ascii')
                    })
            if len(payload) > 0:
                res = self.session.post(f'{self.base_url}/upload', json=payload)
                if not res.ok:
                    raise RuntimeError(res.status_code)



