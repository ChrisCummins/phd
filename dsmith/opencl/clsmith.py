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

from tempfile import NamedTemporaryFile
from labm8 import fs
from collections import namedtuple
from subprocess import Popen, PIPE, STDOUT
from time import time
from pathlib import Path
from typing import Dict, List, Tuple, NewType

import dsmith
from dsmith import Colors

runtime_t = NewType('runtime_t', float)
status_t = NewType('status_t', int)
return_t = namedtuple('return_t', ['runtime', 'status', 'stdout', 'stderr'])

# set these variables to your local CLSmith build:
exec_path = dsmith.data_path("bin", "CLSmith")
cl_launcher_path = dsmith.data_path("bin", "cl_launcher")
include_path = dsmith.data_path("include")

# sanity checks
assert fs.isexe(exec_path)
assert fs.isexe(cl_launcher_path)
assert fs.isfile(fs.path(include_path, "CLSmith.h"))


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
    logging.debug(f"{Colors.BOLD}${Colors.END} " + " ".join(cli))
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


def cl_launcher(program_path: Path, platform_id: int, device_id: int,
                *args, **kwargs) -> return_t:
    """
        Returns:
            return_t: A named tuple consisting of runtime (float),
                status (int), stdout (str), and stderr (str).
    """
    start_time = time()

    cli = cl_launcher_cli(program_path, platform_id, device_id, *args, **kwargs)
    logging.debug(f"{Colors.BOLD}${Colors.END} " + " ".join(cli))
    process = Popen(cli, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    runtime = runtime_t(time() - start_time)

    return return_t(
        runtime=runtime, status=status_t(process.returncode),
        stdout=stdout.decode('utf-8'), stderr=stderr.decode('utf-8'))


def cl_launcher_str(src: str, platform_id: int, device_id: int, timeout: int,
                    *args) -> Tuple[float, int, str, str]:
    """ Invoke cl launcher on source """
    with NamedTemporaryFile(prefix='cl_launcher-', suffix='.cl') as tmp:
        tmp.write(src.encode('utf-8'))
        tmp.flush()

        return cl_launcher(tmp.name, platform_id, device_id, *args,
                           timeout=timeout)


def verify_cl_launcher_run(platform: str, device: str, optimizations: bool,
                           global_size: tuple, local_size: tuple,
                           stderr: str) -> None:
    """ verify that expected params match actual as reported by CLsmith """
    optimizations = "on" if optimizations else "off"

    actual_platform = None
    actual_device = None
    actual_optimizations = None
    actual_global_size = None
    actual_local_size = None
    for line in stderr.split('\n'):
        if line.startswith("Platform: "):
            actual_platform_name = re.sub(r"^Platform: ", "", line).rstrip()
        elif line.startswith("Device: "):
            actual_device_name = re.sub(r"^Device: ", "", line).rstrip()
        elif line.startswith("OpenCL optimizations: "):
            actual_optimizations = re.sub(r"^OpenCL optimizations: ", "", line).rstrip()

        # global size
        match = re.match('^3-D global size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            actual_global_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        match = re.match('^2-D global size \d+ = \[(\d+), (\d+)\]', line)
        if match:
            actual_global_size = (int(match.group(1)), int(match.group(2)), 0)
        match = re.match('^1-D global size \d+ = \[(\d+)\]', line)
        if match:
            actual_global_size = (int(match.group(1)), 0, 0)

        # local size
        match = re.match('^3-D local size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            actual_local_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        match = re.match('^2-D local size \d+ = \[(\d+), (\d+)\]', line)
        if match:
            actual_local_size = (int(match.group(1)), int(match.group(2)), 0)
        match = re.match('^1-D local size \d+ = \[(\d+)\]', line)
        if match:
            actual_local_size = (int(match.group(1)), 0, 0)

        # check if we've collected everything:
        if (actual_platform and actual_device and actual_optimizations and
            actual_global_size and actual_local_size):
            assert(actual_platform == platform)
            assert(actual_device == device)
            assert(actual_optimizations == optimizations)
            assert(actual_global_size == global_size)
            assert(actual_local_size == local_size)
            return
