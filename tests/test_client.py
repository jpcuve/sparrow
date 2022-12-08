from pathlib import Path

from sparrow.client import Client


def test_check(sparrow_client: Client):
    assert sparrow_client.check()


def test_image_upload(sparrow_client: Client):
    folder_pathname = str(Path('.') / 'images')
    sparrow_client.upload(folder_pathname)
