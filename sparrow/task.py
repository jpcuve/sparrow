from time import sleep


def long_running_task():
    for i in range(60):
        print(i)
        sleep(1)
