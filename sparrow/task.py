from time import sleep


def long_running_task(seconds: int):
    for i in range(seconds):
        print(i)
        sleep(1)
