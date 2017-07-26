from labm8 import fs
from collections import namedtuple
from subprocess import Popen, PIPE, STDOUT
from time import time
from typing import Dict, List, Tuple, NewType

runtime_t = NewType('runtime_t', float)
status_t = NewType('status_t', int)
return_t = namedtuple('return_t', ['runtime', 'status', 'stdout', 'stderr'])

# set these variables to your local CLSmith build:
exec_path = fs.path("../lib/CLSmith/build/CLSmith")
cl_launcher_path = fs.path("../lib/CLSmith/build/cl_launcher")
include_path = fs.path("../lib/CLSmith/runtime")


def clsmith_cli(*args, timeout: int=60) -> List[str]:
    return ["timeout", "--signal=9", str(timeout), exec_path] + list(args)


def clsmith(*args) -> return_t:
    """
        Returns:
            return_t: A named tuple consisting of runtime (float),
                status (int), stdout (str), and stderr (str).
    """
    start_time = time()

    cli = clsmith_cli(*args)
    process = Popen(cli, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    runtime = runtime_t(time() - start_time)

    return return_t(
        runtime=runtime, status=status_t(process.returncode),
        stdout=stdout.decode('utf-8'), stderr=stderr.decode('utf-8'))


def cl_launcher_cli(program_path: str, platform_id: int, device_id: int,
                    *args, timeout=60, cl_launcher_path=cl_launcher_path,
                    include_path=include_path) -> str:
    cmd = ["timeout", "--signal=9", str(timeout)] if timeout else []
    return cmd + [cl_launcher_path, '---debug', '-f', program_path,
                  '-p', str(platform_id), '-d', str(device_id),
                  '--include_path', include_path] + list(args)


def cl_launcher(program_path: str, platform_id: int, device_id: int,
                *args, **kwargs) -> return_t:
    """
        Returns:
            return_t: A named tuple consisting of runtime (float),
                status (int), stdout (str), and stderr (str).
    """
    start_time = time()

    cli = cl_launcher_cli(program_path, platform_id, device_id, *args, **kwargs)
    process = Popen(cli, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    runtime = runtime_t(time() - start_time)

    return return_t(
        runtime=runtime, status=status_t(process.returncode),
        stdout=stdout.decode('utf-8'), stderr=stderr.decode('utf-8'))
