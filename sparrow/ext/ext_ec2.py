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

    def find_instances(self) -> List[Dict]:
        instances = self.resource.instances.all()
        with db_sparrow.engine.connect() as conn:
            db_sparrow.save_aws_instances(conn, instances)
            res = db_sparrow.find_aws_instances(conn)
            return res

    def find_available_instance(self) -> str:
        # algorithm as per specs: first find an idle instance, if none available, start a stopped instance
        instances = self.find_instances()
        for instance in instances:
            if instance['state'] == 'running' and instance['job_reference'] is None:
                return instance['id']
        for instance in instances:
            if instance['state'] == 'stopped':
                self.client.start_instances(
                    InstanceIds=[instance['id']],
                    DryRun=True  # check if permissions ok
                )
                # TODO wait for instance to be started?
                return instance['id']



ec2 = Ec2()
