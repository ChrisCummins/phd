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
CLSmith.
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
from dsmith.opencl.db import *
from dsmith.opencl import clsmith


class OpenCLGenerator(Generator):
    """
    Common baseclass for OpenCL program generators.
    """
    # Abstract methods (must be implemented):
    def _generate_one(self, session: session_t) -> Program:
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
            while num_to_generate > 0:
                program = self._generate_one(s)
                s.add(program)
                s.commit()

                # Update progress bar
                num_progs = self.num_programs(s)
                bar.update(num_progs)
                num_to_generate = max_value - num_progs
        print(f"All done! You now have {Colors.BOLD}{num_progs}{Colors.END} "
              "{self} programs in the database")


class CLSmith(OpenCLGenerator):
    __name__ = "clsmith"
    id = Generators.CLSMITH

    def _generate_one(self, session: session_t, depth: int=1) -> Program:
        """ Generate a single CLSmith program. """
        with NamedTemporaryFile(prefix='dsmith-clsmith-', suffix='.cl') as tmp:
            runtime, status, _, stderr = clsmith.clsmith('-o', tmp.name, *flags)

            # A non-zero exit status of clsmith implies that no program was
            # generated. Try again:
            if status:
                if depth > 100:
                    logging.error(stderr)
                    raise OSError(f"Failed to produce {self} program after 100 attempts")
                else:
                    return self._generate_one(session, depth=depth + 1)

            src = fs.read_file(tmp.name)

        # Check if the program is a duplicate. If so, try again:
        sha1 = crypto.sha1_str(src)
        is_dupe = session.query(Program.id)\
            .filter(Program.generator == CLSmith.id,
                    Program.sha1 == sha1).first()
        if is_dupe:
            return self._generate_one(session, depth=depth + 1)

        return Program(
            generator=self.id,
            sha1=crypto.sha1_str(src),
            generation_time=runtime,
            linecount=len(src.split("\n")),
            charcount=len(src),
            src=src)


class DSmith(OpenCLGenerator):
    __name__ = "dsmith"
    id = Generators.DSMITH

    def _generate_one(self, session: session_t) -> Program:
        """ Generate a single program. """
        raise NotImplementedError


class RandChar(OpenCLGenerator):
    """
    This generator produces a uniformly random sequence of ASCII characters, of
    a random length.
    """
    __name__ = "randchar"
    id = Generators.RANDCHAR

    # This is the hardcoded range of kernel lengths found in the GitHub corpus
    # (after preprocessing).
    charcount_range = (33, 451563)

    def _generate_one(self, session: session_t, depth: int=1) -> Program:
        """ Generate a single program. """
        if depth > 100:
            raise OSError(f"Failed to produce {self} program after 100 attempts")

        start_time = time()
        charcount = random.randint(*self.charcount_range)
        src = ''.join(random.choices(string.printable, k=charcount))
        runtime = time() - start_time

        # Check if the program is a duplicate. If so, try again:
        sha1 = crypto.sha1_str(src)
        is_dupe = session.query(Program.id)\
            .filter(Program.generator == CLSmith.id,
                    Program.sha1 == sha1).first()
        if is_dupe:
            return self._generate_one(session, depth=depth + 1)

        return Program(
            generator=self.id,
            sha1=crypto.sha1_str(src),
            generation_time=runtime,
            linecount=len(src.split("\n")),
            charcount=len(src),
            src=src)


class RandTok(OpenCLGenerator):
    __name__ = "randtok"
    id = Generators.RANDTOK

    def _generate_one(self, session: session_t) -> Program:
        """ Generate a single program. """
        raise NotImplementedError
