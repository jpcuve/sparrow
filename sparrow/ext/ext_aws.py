from flask import Flask


class Aws:
    def __init__(self, app: Flask = None):
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        pass

    def start_instance(self, instance_id: str):
        pass

    def stop_instance(self, instance_id: str):
        pass

    def find_available_instance(self) -> int:
        pass


aws = Aws()
