import datetime
import uuid
from typing import List, Dict

from flask import Flask
from sqlalchemy import MetaData, Column, Table, Integer, String, UniqueConstraint, ForeignKey, Text, select, Identity, \
    Float, DateTime

from sparrow import db


class DatabaseSparrow:
    def __init__(self):
        self.engine = None
        self.metadata = MetaData()
        self.users = Table(
            'users', self.metadata,
            Column('id', Integer, nullable=False, primary_key=True),
            Column('username', String(255), nullable=False),
            Column('api_key', String(64), nullable=False),
            UniqueConstraint('api_key')
        )
        self.processes = Table(
            'processes', self.metadata,
            Column('key', String, nullable=False, primary_key=True),
            Column('progress', Float, nullable=False)
        )
        self.aws_instances = Table(
            'aws_instances', self.metadata,
            Column('id', String(32), primary_key=True),  # is the aws instance id
            Column('type', String(32)),
            Column('public_ip_v4', String(16)),
            Column('state', String(32)),
            Column('job_reference', String(64)),  # this is some reference of a busy job on the aws instance
            UniqueConstraint('id')
        )
        self.finetune_jobs = Table(
            'finetune_jobs', self.metadata,
            Column('id', String(36), nullable=False, primary_key=True),
            Column('model_reference', String(255), nullable=False),
            Column('gender', String(16), nullable=False),
            Column('max_train_steps', Integer, nullable=False),
            Column('user_id', None, ForeignKey('users.id')),
            Column('aws_instance_id', None, ForeignKey('aws_instances.id')),
            UniqueConstraint('user_id', 'model_reference')
        )
        self.finetune_job_events = Table(
            'finetune_job_events', self.metadata,
            Column('id', Integer, Identity(), primary_key=True),
            Column('created', DateTime, nullable=False),
            Column('status', String(16), nullable=False),
            Column('progress', Float, nullable=False),  # from 0.0 to 1.0 (>= 1.0 means finished)
            Column('comment', Text),
            Column('finetune_job_id', None, ForeignKey('finetune_jobs.id'))
        )
        self.finetune_job_image_urls = Table(
            'finetune_job_image_urls', self.metadata,
            Column('id', Integer, Identity(), primary_key=True),
            Column('url', String(2048), nullable=False),
            Column('finetune_job_id', None, ForeignKey('finetune_jobs.id')),
            UniqueConstraint('finetune_job_id', 'url')
        )
        self.inference_jobs = Table(
            'inference_jobs', self.metadata,
            Column('id', String(36), nullable=False, primary_key=True),
            Column('prompt', Text),
            Column('negative_prompt', Text),
            Column('num_inference_steps', Integer, nullable=False),
            Column('num_images_per_prompt', Integer, nullable=False),
            Column('guidance_scale', Float, nullable=False),
            Column('aws_instance_id', None, ForeignKey('aws_instances.id')),
            Column('finetune_job_id', None, ForeignKey('finetune_jobs.id'))
        )
        self.inference_job_events = Table(
            'inference_job_events', self.metadata,
            Column('id', Integer, Identity(), primary_key=True),
            Column('created', DateTime, nullable=False),
            Column('status', String(16), nullable=False),
            Column('progress', Float, nullable=False),  # from 0.0 to 1.0 (>= 1.0 means finished)
            Column('comment', Text),
            Column('inference_job_id', None, ForeignKey('inference_jobs.id'))
        )
        self.generated_images = Table(
            'generated_images', self.metadata,
            Column('url', String(2048), nullable=False),
            Column('id', Integer, Identity(), primary_key=True),
            Column('inference_job_id', None, ForeignKey('inference_jobs.id'))
        )

    def init_app(self, app: Flask):
        with app.app_context():
            self.engine = db.get_engine('sparrow')

    def create_all(self):
        self.metadata.create_all(self.engine)

    def get_user_id(self, conn, api_key: str):
        sel_1 = select([self.users.c.id]).where(self.users.c.api_key == api_key)
        rec_1 = conn.execute(sel_1).fetchone()
        return rec_1[0] if rec_1 is not None else None

    def insert_finetune_job(self, conn, user_id: int, model_reference: str, gender: str,
                            max_train_steps: int, image_urls: List[str]) -> int:
        ins_1 = self.finetune_jobs.insert().values(
            id=str(uuid.uuid4()),
            user_id=user_id,
            model_reference=model_reference,
            gender=gender,
            max_train_steps=max_train_steps,
        )
        finetune_job_id = conn.execute(ins_1).inserted_primary_key[0]
        for image_url in image_urls:
            ins_2 = self.finetune_job_image_urls.insert().values(
                finetune_job_id=finetune_job_id,
                url=image_url,
            )
            conn.execute(ins_2)
        return finetune_job_id
    
    def find_finetune_jobs(self, conn, instance_id: str = None) -> List[Dict]:
        sel_1 = select(
            self.finetune_jobs.c.id,
            self.finetune_jobs.c.user_id,
            self.finetune_jobs.c.aws_instance_id,
        )
        if instance_id is not None:
            sel_1.where(self.finetune_jobs.c.aws_instance_id == instance_id)
        res = [{key: rec[index] for index, key in enumerate(['id', 'user_id', 'aws_instance_id'])} 
               for rec in conn.execute(sel_1).fetchall()]
        return res

    def insert_finetune_job_event(self, conn, finetune_job_id, status: str,
                                  progress: float = 0.0, comment: str = None):
        ins_1 = self.finetune_job_events.insert().values(
            created=datetime.datetime.now(),
            finetune_job_id=finetune_job_id,
            status=status,
            progress=progress,
            comment=comment
        )
        conn.execute(ins_1)

    def find_finetune_job_status(self, conn, user_id, finetune_job_id: str) -> str:
        sel_1 = (select(self.finetune_job_events.c.created, self.finetune_job_events.c.status)
                 .where(self.finetune_job_events.c.finetune_job_id == finetune_job_id))
        rec_1 = max(conn.execute(sel_1).fetchall(), key=lambda r: r[0], default=None)
        if rec_1 is None:
            raise RuntimeError("Finetune job not found")
        return rec_1[1]

    def insert_inference_job(self, conn, user_id: int, model_reference: str, prompt: str, negative_prompt: str,
                             num_inference_steps: int, num_images_per_prompt: int, guidance_scale: float) -> int:
        # first, find the corresponding finetune job
        sel_1 = (select([self.finetune_jobs.c.id])
                 .where(self.finetune_jobs.c.user_id == user_id)
                 .where(self.finetune_jobs.c.model_reference == model_reference))
        rec_1 = conn.execute(sel_1).fetchone()
        if rec_1 is None:
            raise RuntimeError("Model not found")
        finetune_job_id = rec_1[0]
        ins_1 = self.inference_jobs.insert().values(
            id=str(uuid.uuid4()),
            finetune_job_id=finetune_job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
        )
        inference_job_id = conn.execute(ins_1).inserted_primary_key[0]
        return inference_job_id
    
    def find_inference_jobs(self, conn, instance_id: str = None) -> List[Dict]:
        sel_1 = select(
            self.inference_jobs.c.id,
            self.inference_jobs.c.user_id,
            self.inference_jobs.c.aws_instance_id,
        )
        if instance_id is not None:
            sel_1.where(self.inference_jobs.c.aws_instance_id == instance_id)
        res = [{key: rec[index] for index, key in enumerate(['id', 'user_id', 'aws_instance_id'])} 
               for rec in conn.execute(sel_1).fetchall()]
        return res

    def insert_inference_job_event(self, conn, inference_job_id, status: str,
                                   progress: float = 0.0, comment: str = None):
        ins_1 = self.inference_job_events.insert().values(
            created=datetime.datetime.now(),
            inference_job_id=inference_job_id,
            status=status,
            progress=progress,
            comment=comment
        )
        conn.execute(ins_1)

    def find_inference_job_status(self, conn, user_id, inference_job_id: str) -> str:
        sel_1 = (select(self.inference_job_events.c.created, self.inference_job_events.c.status)
                 .where(self.inference_job_events.c.inference_job_id == inference_job_id))
        rec_1 = max(conn.execute(sel_1).fetchall(), key=lambda r: r[0], default=None)
        if rec_1 is None:
            raise RuntimeError("Inference job not found")
        return rec_1[1]

    def find_generated_image_urls(self, conn, user_id: int, inference_job_id: int) -> List[str]:
        # ok here we have to be careful that the user cannot not get urls for images that are not his, hence the join
        sel_1 = (select([self.generated_images.c.url])
                 .select_from(self.generated_images
                              .join(self.inference_jobs,
                                    onclause=self.generated_images.c.inference_job_id == self.inference_jobs.c.id)
                              .join(self.finetune_jobs,
                                    onclause=self.inference_jobs.c.finetune_job_id == self.finetune_jobs.c.id))
                 .where(self.generated_images.c.inference_job_id == inference_job_id)
                 .where(self.finetune_jobs.c.user_id == user_id))
        return [rec_1[0] for rec_1 in conn.execute(sel_1).fetchall()]

    def save_aws_instances(self, conn, instances: List):
        sel_1 = (select([self.aws_instances.c.id]))
        aws_instance_ids = [rec_1[0] for rec_1 in conn.execute(sel_1).fetchall()]
        for instance in instances:
            if instance.id in aws_instance_ids:
                upd_1 = (self.aws_instances.update().values(
                    type=instance.instance_type,
                    public_ip_v4=instance.public_ip_address,
                    state=instance.state.get('Name')
                ).where(self.aws_instances.c.id == instance.id))
                conn.execute(upd_1)
            else:
                ins_1 = (self.aws_instances.insert().values(
                    id=instance.id,
                    type=instance.instance_type,
                    public_ip_v4=instance.public_ip_address,
                    state=instance.state.get('Name')
                ))
                conn.execute(ins_1)

    def find_aws_instances(self, conn, instance_id: str = None) -> List[Dict]:
        sel_1 = select(
            self.aws_instances.c.id,
            self.aws_instances.c.type,
            self.aws_instances.c.public_ip_v4,
            self.aws_instances.c.state,
            self.aws_instances.c.job_reference,
        )
        if instance_id is not None:
            sel_1.where(self.aws_instances.c.id == instance_id)
        res = [{key: rec[index] for index, key in enumerate(['id', 'type', 'public_ip_v4', 'state', 'job_reference'])}
               for rec in conn.execute(sel_1).fetchall()]
        return res

    def reference_aws_instance(self, conn, instance_id: str, job_reference: str):
        upd_1 = (self.aws_instances.update().values(job_reference=job_reference)
                 .where(self.aws_instances.c.id == instance_id))
        conn.execute(upd_1)

    def acquire_lock(self, conn, key: str) -> bool:
        sel_1 = (select([self.processes.c.progress]).where(self.processes.c.key == key))
        rec_1 = conn.execute(sel_1).fetchone()
        if rec_1 is None:
            ins_1 = (self.processes.insert().values(key=key, progress=1.0e-10))
            conn.execute(ins_1)
            return True
        elif rec_1[0] == 0:
            upd_1 = (self.processes.update().values(progress=1.0e-10).where(self.processes.c.key == key))
            conn.execute(upd_1)
            return True
        return False

    def release_lock(self, conn, key: str):
        sel_1 = (select([self.processes.c.progress]).where(self.processes.c.key == key))
        rec_1 = conn.execute(sel_1).fetchone()
        if rec_1 is not None:
            upd_1 = (self.processes.update().values(progress=0).where(self.processes.c.key == key))
            conn.execute(upd_1)


db_sparrow = DatabaseSparrow()
