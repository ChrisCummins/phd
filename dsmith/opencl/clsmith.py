#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Interface to CLSmith binaries
"""
import logging

from labm8 import fs
from collections import namedtuple
from subprocess import Popen, PIPE, STDOUT
from time import time
from typing import Dict, List, Tuple, NewType

import dsmith

runtime_t = NewType('runtime_t', float)
status_t = NewType('status_t', int)
return_t = namedtuple('return_t', ['runtime', 'status', 'stdout', 'stderr'])

# set these variables to your local CLSmith build:
exec_path = dsmith.data_path("CLSmith")
cl_launcher_path = fs.path("../lib/clsmith/build/cl_launcher")
include_path = fs.path("../lib/clsmith/runtime")


def clsmith_cli(*args, timeout: int=60, exec_path=exec_path) -> List[str]:
    return ["timeout", "--signal=9", str(timeout), exec_path] + list(args)


def clsmith(*args, exec_path=exec_path) -> return_t:
    """
        Returns:
            return_t: A named tuple consisting of runtime (float),
                status (int), stdout (str), and stderr (str).
    """
    start_time = time()

    cli = clsmith_cli(*args)
    logging.debug("$ " + " ".join(cli))
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
