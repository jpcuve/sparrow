from flask import Flask


class Finetuner:
    def __init__(self, app: Flask = None):
        self.executor = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        pass

    def send_finetuner_job(self):
        pass

    def send_inference_job(self):
        pass


finetuner = Finetuner()
