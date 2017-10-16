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
import cldrive

from labm8 import crypto, fs
from tempfile import NamedTemporaryFile
from sqlalchemy.sql import func
from typing import List

import dsmith
from dsmith.langs import Generator, Harness
from dsmith.opencl import clsmith, generators
from dsmith.opencl.db import *


def _non_zero_threads(session: session_t=None):
    with ReuseSession(session) as s:
        return s.query(Threads)\
                    .filter(Threads.gsize_x > 0).all()


def _zero_threads(session: session_t=None):
    with ReuseSession(session) as s:
        return s.query(Threads)\
                    .filter(Threads.gsize_x == 0).all()


class OpenCLHarness(Harness):
    def make_testcases(self, generator: Generator):
        """ Generate testcases, optionally for a specific generator """
        if generator.__name__ not in self.__generators__:
            raise ValueError(f"incompatible combination {self}:{generator}")

        with Session() as s:
            all_threads = self.all_threads(s)

            for threads in all_threads:
                # Make a list of programs which already have matching testcases
                already_exists = s.query(Program.id)\
                    .join(Testcase)\
                    .filter(Program.generator == generator.id,
                            Testcase.threads_id == threads.id,
                            Testcase.harness == self.id)

                # The list of testcases to make is the compliment of the above:
                todo = s.query(Program)\
                    .filter(Program.generator == generator.id,
                            ~Program.id.in_(already_exists))

                # Determine how many, if any, testcases need to be made:
                nexist = already_exists.count()
                ntodo = todo.count()
                ntotal = nexist + ntodo
                logging.debug(f"{self}:{generator} {threads} testcases = {nexist} / {ntotal}")

                # Break early if there's nothing to do:
                if not ntodo:
                    return

                print(f"Generating {Colors.BOLD}{ntodo}{Colors.END} "
                      f"{self}:{generator} testcases with threads "
                      f"{Colors.BOLD}{threads}{Colors.END}")

                # Bulk insert new testcases:
                s.add_all([
                    Testcase(
                        program_id=program.id,
                        threads_id=threads.id,
                        harness=self.id,
                        input_seed=self.default_seed,
                        timeout=self.default_timeout,
                    ) for program in todo
                ])
                s.commit()

    def testbeds(self, session: session_t=None) -> List[TestbedProxy]:
        with ReuseSession(session) as s:
            if self.id == Harnesses.COMPILE_ONLY:
                q = s.query(Testbed)\
                    .join(Platform)\
                    .filter(Platform.platform == "clang")
                return sorted(TestbedProxy(testbed) for testbed in q)
            else:
                q = s.query(Testbed)\
                    .join(Platform)\
                    .filter(Platform.platform != "clang")
                return sorted(TestbedProxy(testbed) for testbed in q)

    def available_testbeds(self, session: session_t=None) -> List[TestbedProxy]:
        testbeds = []
        with ReuseSession(session) as s:
            for env in cldrive.all_envs():
                testbeds += [TestbedProxy(testbed) for testbed in Testbed.from_env(env, session=s)]

        return sorted(testbeds)


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


class Cl_launcher(OpenCLHarness):
    __name__ = "cl_launcher"
    __generators__ = {
        "clsmith": generators.CLSmith,
    }

    id = Harnesses.CLSMITH
    default_seed = None
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        return _non_zero_threads(session=session)

    def num_results(self, generator: Generator, testbed: str, session: session_t=None):
        with ReuseSession(session) as s:
            testbed_ = Testbed.from_str(testbed, session=s)[0]
            n = s.query(func.count(Result.id))\
                .join(Testcase)\
                .join(Program)\
                .filter(Result.testbed_id == testbed_.id,
                        Program.generator == generator,
                        Testcase.harness == self.id)\
                .scalar()
            return n

    def run(self, testcase: Testcase, testbed: Testbed, session: session_t=None):
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
            logging.info(f"↳  {Colors.BOLD}{return_color}{outcome_name}{Colors.END} "
                         f"after {Colors.BOLD}{runtime:.2f}{Colors.END} seconds")

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


class Cldrive(OpenCLHarness):
    __name__ = "cldrive"
    __generators__ = {
        "dsmith": generators.DSmith,
    }

    id = Harnesses.DSMITH
    default_seed = None
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        return _non_zero_threads(session=session)

    def num_results(self, generator: Generator, testbed: str, session: session_t=None):
        with ReuseSession(session) as s:
            testbed_ = Testbed.from_str(testbed, session=s)[0]
            n = s.query(func.count(Result.id))\
                .join(Testcase)\
                .join(Program)\
                .filter(Result.testbed_id == testbed_.id,
                        Program.generator == generator,
                        Testcase.harness == self.id)\
                .scalar()
            return n


class Clang(OpenCLHarness):
    __name__ = "clang"
    __generators__ = {
        "clsmith": generators.CLSmith,
        "dsmith": generators.DSmith,
    }

    id = Harnesses.COMPILE_ONLY
    default_seed = None
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        return _zero_threads(session=session)

    def num_results(self, generator: Generator, testbed: str, session: session_t=None):
        with ReuseSession(session) as s:
            testbed_ = Testbed.from_str(testbed, session=s)[0]
            n = s.query(func.count(Result.id))\
                .join(Testcase)\
                .join(Program)\
                .filter(Result.testbed_id == testbed_.id,
                        Program.generator == generator,
                        Testcase.harness == self.id)\
                .scalar()
            return n
