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
import logging
import math
from sqlalchemy.sql import func

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

    @property
    def testbeds(self):
        with Session() as s:
            return s.query(Testbed)

    @property
    def available_testbeds(self):
        import cldrive

        with Session() as s:
            testbeds = []

            for env in cldrive.all_envs():
                testbeds += Testbed.from_env(env, session=s)

            # not "yield from", since we need the session to hang around
            yield from testbeds
