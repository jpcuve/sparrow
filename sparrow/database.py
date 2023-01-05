from typing import List

from flask import Flask
from sqlalchemy import MetaData, Column, Table, Integer, String, UniqueConstraint, ForeignKey, Text, select, Identity, \
    Float

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
            Column('id', Integer, Identity(), primary_key=True),
            Column('aws_instance_id', String(32), nullable=False),
            Column('status', String(16), nullable=False),
            UniqueConstraint('aws_instance_id')
        )
        self.finetune_jobs = Table(
            'finetune_jobs', self.metadata,
            Column('id', Integer, Identity(), primary_key=True),
            Column('model_reference', String(255), nullable=False),
            Column('gender', String(16), nullable=False),
            Column('max_train_steps', Integer, nullable=False),
            Column('status', String(16), nullable=False),
            Column('user_id', None, ForeignKey('users.id')),
            UniqueConstraint('user_id', 'model_reference')
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
            Column('id', Integer, Identity(), primary_key=True),
            Column('prompt', Text),
            Column('negative_prompt', Text),
            # gender??? not the same as finetune?
            Column('num_inference_steps', Integer, nullable=False),
            Column('num_images_per_prompt', Integer, nullable=False),
            Column('guidance_scale', Float, nullable=False),
            Column('status', String(16), nullable=False),
            Column('finetune_job_id', None, ForeignKey('finetune_jobs.id'))
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

    def insert_finetune_job(self, conn, user_id: int, model_reference: str, image_urls: List[str]) -> int:
        ins_1 = self.finetune_jobs.insert().values(
            user_id=user_id,
            model_reference=model_reference,
            gender='man',  # TODO
            max_train_steps=5000,  # TODO
            status='SUBMITTED',  # TODO
        )
        finetune_job_id = conn.execute(ins_1).inserted_primary_key[0]
        for image_url in image_urls:
            ins_2 = self.finetune_job_image_urls.insert().values(
                finetune_job_id=finetune_job_id,
                url=image_url,
            )
            conn.execute(ins_2)
        return finetune_job_id
    
    def find_finetune_job_status(self, conn, user_id, finetune_job_id: int) -> str:
        sel_1 = (select([self.finetune_jobs.c.status]).where(self.finetune_jobs.c.id == finetune_job_id))
        rec_1 = conn.execute(sel_1).fetchone()
        if rec_1 is None:
            raise RuntimeError("Finetune job not found")
        return rec_1[0]

    def insert_inference_job(self, conn, user_id: int, model_reference: str, prompt: str, negative_prompt: str) -> int:
        # first, find the corresponding finetune job
        sel_1 = (select([self.finetune_jobs.c.id])
                 .where(self.finetune_jobs.c.user_id == user_id)
                 .where(self.finetune_jobs.c.model_reference == model_reference))
        rec_1 = conn.execute(sel_1).fetchone()
        if rec_1 is None:
            raise RuntimeError("Model not found")
        finetune_job_id = rec_1[0]
        ins_1 = self.inference_jobs.insert().values(
            finetune_job_id=finetune_job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=150,  # TODO
            num_images_per_prompt=6,  # TODO
            guidance_scale=6.5,  # TODO
            status='SUBMITTED',  # TODO
        )
        inference_job_id = conn.execute(ins_1).inserted_primary_key[0]
        return inference_job_id
    
    def find_inference_job_status(self, conn, user_id, inference_job_id: int) -> str:
        sel_1 = (select([self.inference_jobs.c.status]).where(self.inference_jobs.c.id == inference_job_id))
        rec_1 = conn.execute(sel_1).fetchone()
        if rec_1 is None:
            raise RuntimeError("Inference job not found")
        return rec_1[0]

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
