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
OpenCL test harnesses.
"""
from labm8 import crypto, fs
from tempfile import NamedTemporaryFile


import dsmith
from dsmith.langs import Harness
from dsmith.opencl import clsmith
from dsmith.opencl.db import *


def _cl_launcher(src: str, platform_id: int, device_id: int, timeout: int,
                 *args) -> Tuple[float, int, str, str]:
    """ Invoke cl launcher on source """
    with NamedTemporaryFile(prefix='cl_launcher-', suffix='.cl') as tmp:
        tmp.write(src.encode('utf-8'))
        tmp.flush()

        return clsmith.cl_launcher(tmp.name, platform_id, device_id, *args,
                                   timeout=timeout)


def _verify_cl_launcher_run(platform: str, device: str, optimizations: bool,
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


class Cl_launcher(Harness):
    __name__ = "cl_launcher"
    id = Harnesses.CLSMITH
    default_seed = None
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        with ReuseSession(session) as s:
            return s.query(Threads)\
                .filter(Threads.gsize_x > 0).all()

    def run(self, testbed, testcase, session: session_t=None):
        """ execute a testcase """
        with ReuseSession(session) as s:
            platform_id, device_id = testbed.ids

            runtime, returncode, stdout, stderr = _cl_launcher(
                    testcase.program.src, platform_id, device_id,
                    testcase.timeout)

            # assert that executed params match expected
            _verify_cl_launcher_run(platform=testbed.platform.platform,
                                    device=testbed.platform.device,
                                    optimizations=testbed.optimizations,
                                    global_size=testcase.threads.gsize,
                                    local_size=testcase.threads.lsize,
                                    stderr=stderr)

            # create new result
            stdout_ = stdout  # util.escape_stdout(stdout)
            stdout = get_or_create(
                s, Stdout,
                sha1=crypto.sha1_str(stdout_),
                stdout=stdout_)

            stderr_ = stderr  # util.escape_stderr(stderr)
            stderr = get_or_create(
                s, Stderr,
                sha1=crypto.sha1_str(stderr_),
                stderr=stderr_)
            session.flush()  # required to get IDs

            # determine result outcome
            outcome = ClsmithResult.get_outcome(
                returncode, stderr_, runtime, testcase.timeout)

            outcome_name = Outcomes.to_str(outcome)
            return_color = Colors.RED if returncode else Colors.GREEN
            logging.debug(f"↳  {Colors.BOLD}{return_color}{returncode} "
                          f"{outcome_name}{Colors.END} after "
                          f"{Colors.BOLD}{runtime:.2f}{Colors.END} seconds")

            result = ClsmithResult(
                testbed_id=testbed.id,
                testcase_id=testcase.id,
                returncode=returncode,
                outcome=outcome,
                runtime=runtime,
                stdout_id=stdout.id,
                stderr_id=stderr.id)

            session.add(result)
            session.commit()


class Cldrive(Harness):
    __name__ = "cldrive"
    id = Harnesses.DSMITH
    default_seed = None
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        with ReuseSession(session) as s:
            return s.query(Threads)\
                .filter(Threads.gsize_x > 0).all()


# class Clang(Harness):
#     __name__ = "clang"
#     id = Harnesses.COMPILE_ONLY
#     default_seed = None
#     default_timeout = 60

#     def all_threads(self, session: session_t=None):
#         with ReuseSession(session) as s:
#             return s.query(Threads)\
#                 .filter(Threads.gsize_x == 0).all()
