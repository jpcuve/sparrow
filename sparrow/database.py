import json
import uuid
from typing import List, Dict

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
        self.images = Table(
            'images', self.metadata,
            Column('id', Integer, Identity(), primary_key=True),
            Column('prompts', Text, nullable=False),
            Column('data', Text),
            Column('user_id', None, ForeignKey('users.id')),
        )
        self.training_requests = Table(
            'training_requests', self.metadata,
            Column('id', String(64), nullable=False, primary_key=True),
            Column('parameters', Text),
            Column('completed', Integer, nullable=False),
            Column('user_id', None, ForeignKey('users.id')),
        )
        self.inference_requests = Table(
            'inference_requests', self.metadata,
            Column('id', Integer, nullable=False, primary_key=True),
            Column('training_request_id', None, ForeignKey('training_requests.id'))
        )
        self.processes = Table(
            'processes', self.metadata,
            Column('key', String, nullable=False, primary_key=True),
            Column('progress', Float, nullable=False)
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

    def insert_image(self, conn, user_id: int, prompts: List[str], data: str) -> int:
        ins_1 = self.images.insert().values(
            user_id=user_id,
            prompts='|'.join(prompts),
            data=data,
        )
        image_id = conn.execute(ins_1).inserted_primary_key[0]
        return image_id

    def insert_training_request(self, conn, user_id: int, parameters: Dict) -> str:
        train_id = str(uuid.uuid4())
        ins_1 = self.training_requests.insert().values(
            id=train_id,
            user_id=user_id,
            parameters=json.dumps(parameters),
            completed=0,
        )
        conn.execute(ins_1)
        return train_id

    def get_training_status(self, conn, user_id: int, train_id: str) -> int:
        sel_1 = (select([self.training_requests.c.completed])
                 .where(self.training_requests.c.id == train_id)
                 .where(self.training_requests.c.user_id == user_id))
        rec_1 = conn.execute(sel_1).fetchone()
        return rec_1[0] if rec_1 is not None else 0

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
