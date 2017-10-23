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
Solidity program generators.
"""
import humanize
import logging
import random
import string
import progressbar
import math

from time import time
from labm8 import crypto, fs
from sqlalchemy.sql import func
from tempfile import NamedTemporaryFile

import dsmith.opencl.db

from dsmith import Colors
from dsmith.langs import Generator
from dsmith.sol.db import *


class SolidityGenerator(Generator):
    """
    Common baseclass for program generators.
    """
    # Abstract methods (must be implemented):
    def generate_one(self, session: session_t) -> ProgramProxy:
        """ Generate a single program. """
        raise NotImplementedError("abstract class")

    # Default methods (may be overriden):

    def __repr__(self):
        return f"{Colors.BOLD}{Colors.GREEN}{self.__name__}{Colors.END}"

    def num_programs(self, session: session_t=None) -> int:
        """ return the number of generated programs in the database """
        with ReuseSession(session) as s:
            return s.query(func.count(Program.id))\
                .filter(Program.generator == self.id)\
                .scalar()

    def sloc_total(self, session: session_t=None) -> int:
        """ return the total linecount of generated programs """
        with ReuseSession(session) as s:
            return s.query(func.sum(Program.linecount))\
                .filter(Program.generator == self.id)\
                .scalar()

    def generation_time(self, session: session_t=None) -> float:
        """ return the total generation time of all programs """
        with ReuseSession(session) as s:
            return s.query(func.sum(Program.generation_time))\
                .filter(Program.generator == self.id)\
                .scalar() or 0

    def num_testcases(self, session: session_t=None) -> int:
        """ return the total number of testcases """
        with ReuseSession(session) as s:
            return s.query(func.count(Testcase.id))\
                .join(Program)\
                .filter(Program.generator == self.id)\
                .scalar()

    def generate(self, n: int=math.inf, up_to: int=math.inf) -> None:
        """ generate 'n' new programs 'up_to' this many exist in db """
        with Session() as s:
            num_progs = self.num_programs(s)

            # Determine the termination criteria:
            if n == math.inf and up_to == math.inf:
                max_value = math.inf
                bar_max = progressbar.UnknownLength
            elif n == math.inf:
                max_value = up_to
                bar_max = max_value
            else:
                max_value = num_progs + n
                bar_max = max_value

            # Exit early if possible:
            if num_progs >= max_value:
                print(f"There are already {Colors.BOLD}{num_progs}{Colors.END} "
                      "programs in the database. Nothing to be done.")
                return

            # Print a preamble message:
            num_to_generate = max_value - num_progs
            if num_to_generate < math.inf:
                estimated_time = (self.generation_time(s) / max(num_progs, 1)) * num_to_generate
                eta = humanize.naturaldelta(datetime.timedelta(seconds=estimated_time))
                print(f"{Colors.BOLD}{num_to_generate}{Colors.END} programs are "
                      "to be generated. Estimated generation time is " +
                      f"{Colors.BOLD}{eta}{Colors.END}.")
            else:
                print(f"Generating programs {Colors.BOLD}forever{Colors.END} ...")

            bar = progressbar.ProgressBar(initial_value=num_progs,
                                          max_value=bar_max,
                                          redirect_stdout=True)

            # The actual generation loop:
            buf = []
            while num_progs < max_value:
                buf.append(self.generate_one(s))

                # Update progress bar
                num_progs += 1
                bar.update(num_progs)

                if len(buf) >= dsmith.DB_BUF_SIZE:
                    save_proxies(s, buf)
                    num_progs = self.num_programs(s)
                    buf = []
            save_proxies(s, buf)
        print(f"All done! You now have {Colors.BOLD}{num_progs}{Colors.END} "
              "{self} programs in the database")


class RandChar(SolidityGenerator):
    """
    This generator produces a uniformly random sequence of ASCII characters, of
    a random length.
    """
    __name__ = "randchar"
    id = Generators.RANDCHAR

    # Arbitrary range
    charcount_range = (100, 100000)

    def generate_one(self, session: session_t) -> ProgramProxy:
        """ Generate a single program. """
        start_time = time()
        charcount = random.randint(*self.charcount_range)
        src = ''.join(random.choices(string.printable, k=charcount))
        runtime = time() - start_time

        return ProgramProxy(generator=self.id, generation_time=runtime,
                            src=src)
