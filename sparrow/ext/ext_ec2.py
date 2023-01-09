from typing import List, Dict

from flask import Flask
import boto3

from sparrow.database import db_sparrow


class Ec2:
    def __init__(self, app: Flask = None):
        session = boto3.Session(profile_name='hexo')
        self.client = session.client('ec2')
        self.resource = session.resource('ec2')
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

    def find_instances(self) -> List[Dict]:
        instances = self.resource.instances.all()
        with db_sparrow.engine.connect() as conn:
            db_sparrow.save_aws_instances(conn, instances)
            res = db_sparrow.find_aws_instances(conn)
            return res

    def find_available_instance(self) -> str:
        # algorithm as per specs: first find an idle instance, if none available, start a stopped instance
        # to find an idle instance, I scan all the jobs associated to the instance and verify they are TERMINATED
        instances = self.find_instances()
        for instance in instances:
            if instance['state'] == 'running':

                pass
        for instance in instances:
            if instance['state'] == 'stopped':
                # start instance
                return instance['id']


ec2 = Ec2()
