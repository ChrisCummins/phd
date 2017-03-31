from labm8 import fs
from subprocess import Popen, PIPE, STDOUT
from time import time


# set these variables to your local CLSmith build:
exec_path = fs.path("~/src/CLSmith/build/CLSmith")
cl_launcher_path = fs.path("~/src/CLSmith/build/cl_launcher")
include_path = fs.path("~/src/CLSmith/runtime")


def clsmith_cli(*args, timeout=30):
    return ["timeout", "--signal=9", str(timeout), exec_path] + list(args)


def clsmith(*args):
    """
        Returns:
            (float, int, str, str): Runtime, status, stdout, and stderr.
    """
    start_time = time()

    cli = clsmith_cli(*args)
    process = Popen(cli, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    runtime = time() - start_time

    return (runtime, process.returncode,
            stdout.decode('utf-8'), stderr.decode('utf-8'))


def cl_launcher_cli(program_path: str, platform_id: int, device_id: int,
                    *args, timeout=30) -> str:
    return ["timeout", "--signal=9", str(timeout),
            cl_launcher_path, '---debug', '-f', program_path,
            '-p', str(platform_id), '-d', str(device_id),
            '--include_path', include_path] + list(args)


def cl_launcher(program_path: str, platform_id: int, device_id: int, *args):
    """
        Returns:
            (float, int, stdout, stderr): Runtime, status, stdout, and stderr.
    """
    start_time = time()

    cli = cl_launcher_cli(program_path, platform_id, device_id, *args)
    process = Popen(cli, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    runtime = time() - start_time

    return (runtime, process.returncode,
            stdout.decode('utf-8'), stderr.decode('utf-8'))
