from typing import List, Dict

from flask import Flask
import boto3

class Ec2:
    def __init__(self, app: Flask = None):
        self.client = boto3.client('ec2')
        self.resource = boto3.resource('ec2')
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        pass

    def start_instance(self, instance_id: str):
        self.client.start_instances(
            InstanceIds=[instance_id],
            DryRun=True
        )

    def stop_instance(self, instance_id: str):
        self.client.stop_instances(
            InstanceIds=[instance_id],
            DryRun=True
        )

    def find_available_instance(self) -> List[Dict]:
        return [{
            'id': instance.id,
            'platform': instance.platform,
            'instance_type': instance.instance_type,
            'public_ip_address': instance.public_ip_address,
            'ami': instance.image.id,
            'state': instance.state,
        } for instance in self.resource.instances.all()]


ec2 = Ec2()
