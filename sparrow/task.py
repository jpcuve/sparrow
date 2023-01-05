from time import sleep


def long_running_task():
    for i in range(10):
        print(i)
        sleep(1)
