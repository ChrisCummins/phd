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

    def mkgenerator(self, name: str) -> Generator:
        """ Instantiate generator from string """
        generator = self.__generators__.get(name)
        if not generator:
            raise ValueError("Unknown generator")
        return generator()

    def mktestcases(self, generator: Generator=None) -> None:
        """ Generate testcases, optionally for a specific generator """
        with Session(commit=True) as s:
            all_threads = s.query(Threads).all()

            print(all_threads)
            while True:
                programs = s.query(Program.id)
                testcases = s.query(Testcase)
                # if generator:
                #     programs = programs\
                #         .filter(Program.generator == generator.generator_t)
                #     testcases = testcases\
                #         .join(Program)\
                #         .filter(Program.generator == generator.generator_t)
                bar_max = programs.count() * len(all_threads)

                for threads in all_threads:
                    already_exists = s.query(Program.id)\
                        .join(Testcase)\
                        .filter(Testcase.threads_id == threads.id)

                    todo = s.query(Program)\
                        .filter(~Program.id.in_(already_exists))
                    print(threads, todo.count())

                    harness = TODO
                    seed = SEED
                    timeout = TIMEOUT

                    testcases = [
                        Testcase(
                            program=program,
                            threads=threads,
                            harness=harness,
                            input_seed=seed,
                            timeout=timeout,
                        ) for program in todo
                    ]
                return
                # print(testcases.count(), bar_max, testcases.count() / bar_max)

    # program_id = sql.Column(Program.id_t, sql.ForeignKey("programs.id"), nullable=False)
    # threads_id = sql.Column(Threads.id_t, sql.ForeignKey("threads.id"), nullable=False)
    # harness = sql.Column(Harnesses.column_t, nullable=False)
    # input_seed = sql.Column(sql.Integer, nullable=False)
    # timeout = sql.Column(sql.Integer, nullable=False)
