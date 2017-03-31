from labm8 import fs
from subprocess import Popen, PIPE, STDOUT
from time import time


exec_path = fs.path("~/src/CLSmith/build/CLSmith")
include_path = fs.path("~/src/CLSmith/runtime")


def clsmith(*args):
    start_time = time()

    cli = [exec_path] + list(args)
    process = Popen(cli, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    runtime = time() - start_time

    return runtime, stdout.decode('utf-8'), stderr.decode('utf-8')
