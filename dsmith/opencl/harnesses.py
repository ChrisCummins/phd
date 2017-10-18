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
import subprocess
import progressbar
import sys

from sqlalchemy.sql import func
from tempfile import NamedTemporaryFile
from time import time
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


def _log_outcome(outcome: Outcomes, runtime: float):
    """ verbose logging output """
    outcome_name = Outcomes.to_str(outcome)
    return_color = Colors.GREEN if outcome == Outcomes.PASS else Colors.RED
    logging.info(f"↳  {Colors.BOLD}{return_color}{outcome_name}{Colors.END} "
                 f"after {Colors.BOLD}{runtime:.2f}{Colors.END} seconds")


class OpenCLHarness(Harness):
    """ Common superclass for OpenCL test harnesses """

    def run(self, session: session_t, testcase: Testcase, testbed: Testbed):
        """ execute a testcase """
        raise NotImplementedError

    def make_testcases(self, generator: Generator):
        """ Generate testcases, optionally for a specific generator """
        # Sanity check
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
                todo = s.query(Program.id)\
                    .filter(Program.generator == generator.id,
                            ~Program.id.in_(already_exists))

                # Determine how many, if any, testcases need to be made:
                ndone = already_exists.count()
                ntodo = todo.count()
                ntotal = ndone + ntodo
                logging.debug(f"{self}:{generator} {threads} testcases = {ndone} / {ntotal}")

                # Break early if there's nothing to do:
                if not ntodo:
                    return

                print(f"Generating {Colors.BOLD}{ntodo}{Colors.END} "
                      f"{self}:{generator} testcases with threads "
                      f"{Colors.BOLD}{threads}{Colors.END}")

                # Bulk insert new testcases:
                s.add_all(Testcase(
                    program_id=program.id,
                    threads_id=threads.id,
                    harness=self.id,
                    input_seed=self.default_seed,
                    timeout=self.default_timeout,
                ) for program in todo)
                s.commit()


    def testbeds(self, session: session_t=None) -> List[TestbedProxy]:
        with ReuseSession(session) as s:
            if self.id == Harnesses.CLANG:
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


class Cl_launcher(OpenCLHarness):
    """
    cl_launcher, the CLSmith test harness.
    """
    __name__ = "cl_launcher"
    __generators__ = {
        "clsmith": generators.CLSmith,
    }

    id = Harnesses.CL_LAUNCHER
    default_seed = None
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        return _non_zero_threads(session=session)

    def run(self, session: session_t, testcase: Testcase, testbed: Testbed):
        """ execute a testcase using cl_launcher """
        # run testcase
        platform_id, device_id = testbed.ids
        runtime, returncode, stdout_, stderr_ = clsmith.cl_launcher_str(
                testcase.program.src, platform_id, device_id,
                testcase.timeout)

        # assert that executed params match expected
        clsmith.verify_cl_launcher_run(platform=testbed.platform.platform,
                                       device=testbed.platform.device,
                                       optimizations=testbed.optimizations,
                                       global_size=testcase.threads.gsize,
                                       local_size=testcase.threads.lsize,
                                       stderr=stderr_)

        # stdout / stderr
        stdout = Stdout.from_str(session, stdout_)
        stderr = Stderr.from_str(session, stderr_)
        session.flush()  # required to get IDs

        # outcome
        outcome = Cl_launcherResult.get_outcome(
            returncode, stderr_, runtime, testcase.timeout)
        _log_outcome(outcome, runtime)

        # create result
        result = Cl_launcherResult(
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
    """
    cldrive test harness.
    """
    __name__ = "cldrive"
    __generators__ = {
        "dsmith": generators.DSmith,
    }

    id = Harnesses.CLDRIVE
    default_seed = None
    default_timeout = 60

    def all_threads(self, session: session_t=None):
        return _non_zero_threads(session=session)


class Clang(OpenCLHarness):
    """
    Frontend-only clang harness.
    """
    __name__ = "clang"
    __generators__ = {
        "clsmith": generators.CLSmith,
        "dsmith": generators.DSmith,
        "randchar": generators.RandChar,
        "randtok": generators.RandTok,
    }
    __clangs__ = {
        "3.6.2": dsmith.root_path("lib", "llvm", "3.6.2", "build", "bin", "clang"),
        "3.7.1": dsmith.root_path("lib", "llvm", "3.7.1", "build", "bin", "clang"),
        "3.8.1": dsmith.root_path("lib", "llvm", "3.8.1", "build", "bin", "clang"),
        "3.9.1": dsmith.root_path("lib", "llvm", "3.9.1", "build", "bin", "clang"),
        "4.0.1": dsmith.root_path("lib", "llvm", "4.0.1", "build", "bin", "clang"),
        "5.0.0": dsmith.root_path("lib", "llvm", "5.0.0", "build", "bin", "clang"),
        "trunk": dsmith.root_path("lib", "llvm", "trunk", "build", "bin", "clang"),
    }

    id = Harnesses.CLANG
    default_seed = None
    default_timeout = 60

    _clsmith_header = fs.read_file(dsmith.data_path("include", "clsmith.h"))

    def all_threads(self, session: session_t=None):
        return _zero_threads(session=session)

    def _inline_clsmith_header(self, program: Program) -> str:
        """ inline CLSmith header if required and return src """
        if program.generator == Generators.CLSMITH:
            pre, post = program.src.split('#include "CLSmith.h"')
            return pre + self._clsmith_header + post
        else:
            return program.src

    def run(self, session: session_t, testcase: Testcase, testbed: Testbed):
        """ compile a testcase using clang frontend """
        # Sanity checks
        if not testbed.optimizations:
            raise ValueError("Clang testbed requires optimizations enables")
        if testbed.platform.platform != "clang":
            raise ValueError(f"Platform {testbed.platform} is not clang")

        clang = self.__clangs__.get(testbed.platform.driver)
        if not clang:
            raise LookupError(f"clang version '{clang}' does not exist")
        if not fs.isexe(clang):
            raise OSError(f"clang binary '{clang}' not found")

        # Run clang frontend
        with NamedTemporaryFile(prefix='dsmith-clang-', delete=False) as tmpfile:
            src_path = tmpfile.name
        try:
            # Dump source code to file
            with open(src_path, "w") as outfile:
                print(self._inline_clsmith_header(testcase.program), file=outfile)

            # Construct compile command
            cmd = ['timeout', '-s9', str(testcase.timeout), clang, '-cc1', '-xcl', src_path]

            logging.debug(f"{Colors.BOLD}${Colors.END} " + " ".join(cmd))

            start_time = time()
            process = subprocess.Popen(
                cmd, universal_newlines=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr_ = process.communicate()

            returncode = process.returncode
            runtime = time() - start_time
            stderr_ = stderr_.strip()
        finally:
            fs.rm(src_path)

        # stdout / stderr
        stdout = Stdout.from_str(session, "")
        stderr = Stderr.from_str(session, stderr_)
        session.flush()  # required to get IDs

        # outcome
        outcome = Cl_launcherResult.get_outcome(
            returncode, stderr_, runtime, testcase.timeout)
        _log_outcome(outcome, runtime)

        # create result
        result = Cl_launcherResult(
            testbed_id=testbed.id,
            testcase_id=testcase.id,
            returncode=returncode,
            outcome=outcome,
            runtime=runtime,
            stdout_id=stdout.id,
            stderr_id=stderr.id)
        session.add(result)
        session.commit()
