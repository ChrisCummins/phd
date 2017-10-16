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
The OpenCL programming language.
"""
import cldrive
import datetime
import humanize
import logging
import math
import progressbar
import re
import sys

from sqlalchemy.sql import func
from typing import List

import dsmith
import dsmith.opencl.db

from dsmith import Colors
from dsmith.langs import Harness, Generator, Language
from dsmith.opencl.db import *
from dsmith.opencl.generators import CLSmith, DSmith


class OpenCL(Language):
    __name__ = "opencl"

    __generators__ = {
        None: DSmith,
        "dsmith": DSmith,
        "clsmith": CLSmith,
    }

    def __init__(self):
        if db.engine is None:
            db.init()


    def mktestcases(self, generator: Generator, harness: Harness,
                    session: session_t=None) -> None:
        """ Generate testcases, optionally for a specific generator """
        with ReuseSession(session) as s:
            all_threads = harness.all_threads(s)

            for threads in all_threads:
                # Make a list of programs which already have matching testcases
                already_exists = s.query(Program.id)\
                    .join(Testcase)\
                    .filter(Program.generator == generator.generator_t,
                            Testcase.threads_id == threads.id,
                            Testcase.harness == harness.id)

                # The list of testcases to make is the compliment of the above:
                todo = s.query(Program)\
                    .filter(Program.generator == generator.generator_t,
                            ~Program.id.in_(already_exists))

                # Determine how many, if any, testcases need to be made:
                nexist = already_exists.count()
                ntodo = todo.count()
                ntotal = nexist + ntodo
                logging.debug(f"{generator.__name__}:{harness.__name__} {threads} testcases = {nexist} / {ntotal}")

                # Break early if there's nothing to do:
                if not ntodo:
                    return

                print(f"Generating {Colors.BOLD}{ntodo}{Colors.END} "
                      f"{Colors.BOLD}{Colors.GREEN}{generator.__name__}{Colors.END}:"
                      f"{Colors.BOLD}{Colors.YELLOW}{harness.__name__}{Colors.END} "
                      "testcases with threads "
                      f"{Colors.BOLD}{threads}{Colors.END}")

                # Bulk insert new testcases:
                s.add_all([
                    Testcase(
                        program_id=program.id,
                        threads_id=threads.id,
                        harness=harness.id,
                        input_seed=harness.default_seed,
                        timeout=harness.default_timeout,
                    ) for program in todo
                ])
                s.commit()

    def mktestbeds(self, string: str) -> List[Testbed]:
        """ Instantiate testbed(s) by name """
        with Session() as s:
            return [str(testbed) for testbed in Testbed.from_str(string, session=s)]


    def _run_testcases(self, testbed: Testbed, generator: Generator,
                       harness: Harness, session: session_t=None):
        with ReuseSession(session) as s:
            already_done = s.query(Result.testcase_id)\
                .join(Testcase)\
                .join(Program)\
                .filter(Result.testbed_id == testbed.id,
                        Testcase.harness == harness.id,
                        Program.generator == generator.id)

            todo = s.query(Testcase)\
                .join(Program)\
                .filter(Testcase.harness == harness.id,
                        Program.generator == generator.id,
                        ~Testcase.id.in_(already_done))

            ndone = already_done.count()
            ntodo = todo.count()

            # Break early if we have nothing to do
            if not ntodo:
                return

            runtime = s.query(func.sum(Result.runtime))\
                .join(Testcase)\
                .join(Program)\
                .filter(Result.testbed_id == testbed.id,
                        Testcase.harness == harness.id,
                        Program.generator == generator.id)\
                .scalar()

            estimated_time = (runtime / ndone) * ntodo
            eta = humanize.naturaldelta(datetime.timedelta(seconds=estimated_time))

            words_ntodo = humanize.intcomma(ntodo)
            print(f"Running {Colors.BOLD}{words_ntodo} "
                  f"{Colors.GREEN}{generator}{Colors.END}"
                  f":{Colors.BOLD}{Colors.YELLOW}{harness}{Colors.END} "
                  "testcases "
                  f"on {Colors.BOLD}{Colors.PURPLE}{testbed.num}{Colors.END}. "
                  f"Estimated runtime is {Colors.BOLD}{eta}{Colors.END}.")

            bar = progressbar.ProgressBar(initial_value=ndone,
                                          max_value=ndone + ntodo,
                                          redirect_stdout=True)

            for testcase in todo:
                harness.run(testbed, testcase, s)
                ndone = already_done.count()
                bar.update(ndone)

    def run_testcases(self, testbeds: List[str],
                      pairs: List[Tuple[Generator, Harness]]) -> None:
        with Session() as s:
            for generator, harness in pairs:
                for testbed_name in testbeds:
                    testbed = Testbed.from_str(testbed_name, session=s)[0]
                    self._run_testcases(testbed, generator, harness, s)

    def describe_testbeds(self, file=sys.stdout) -> None:
        print(f"The following {self} testbeds are in the database:", file=file)
        with Session() as s:
            for testbed in self.testbeds(session=s):
                testbed_ = Testbed.from_str(testbed, session=s)[0]
                print("    ", testbed, testbed_.platform, file=file)

            print(f"\nThe following {self} testbeds are available on this machine:",
                  file=file)
            for testbed in self.available_testbeds(session=s):
                testbed_ = Testbed.from_str(testbed, session=s)[0]
                print("    ", testbed, testbed_.platform, file=file)

    def describe_results(self, file=sys.stdout) -> None:
        with Session() as s:
            for generator in self.generators:
                for harness in generator.harnesses:
                    for testbed in self.testbeds(session=s):
                        num_results = harness.num_results(generator, testbed)
                        if num_results:
                            word_num = humanize.intcomma(num_results)
                            print(f"There are {Colors.BOLD}{word_num}{Colors.END} "
                                  f"{generator}:{harness} "
                                  f"results on {testbed}.", file=file)

    def testbeds(self, session: session_t=None):
        """ Return all testbeds in data store """
        with ReuseSession(session) as s:
            return sorted([str(testbed) for testbed in s.query(Testbed)])

    def available_testbeds(self, session: session_t=None):
        """ Return all testbeds on the current machine """
        testbeds = []
        with ReuseSession(session) as s:
            for env in cldrive.all_envs():
                testbeds += [str(testbed) for testbed in Testbed.from_env(env, session=s)]

        return sorted(testbeds)
