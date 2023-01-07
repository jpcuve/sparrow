from time import sleep

from sparrow.database import db_sparrow
from sparrow.ext.ext_ec2 import ec2


def long_running_task(seconds: int):
    for i in range(seconds):
        print(i)
        sleep(1)
