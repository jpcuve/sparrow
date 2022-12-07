from typing import List

from flask import Flask
from sqlalchemy import MetaData, Column, Table, Integer, String, UniqueConstraint, ForeignKey, Text, select

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
        self.training_requests = Table(
            'training_requests', self.metadata,
            Column('id', String(64), nullable=False, primary_key=True),
            Column('parameters', Text),
            Column('completed', Integer, nullable=False),
            Column('user_id', None, ForeignKey('users.id')),
        )
        self.inference_requests = Table(
            'inference_requests', self.metadata,
            Column('id', Integer, nullable=False, primary_key=True)
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


db_sparrow = DatabaseSparrow()
