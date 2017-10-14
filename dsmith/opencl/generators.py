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
import progressbar
import math

from labm8 import crypto, fs
from sqlalchemy.sql import func
from tempfile import NamedTemporaryFile

import dsmith.opencl.db

from dsmith import Colors
from dsmith.langs import Generator
from dsmith.opencl.db import *
from dsmith.opencl import clsmith
from dsmith.opencl.harnesses import Clang, Cldrive, Cl_launcher


def _make_clsmith_program(session: session_t, *flags, depth=1) -> None:
    """
    Arguments:
        *flags: Additional flags to CLSmith.
    """
    with NamedTemporaryFile(prefix='clsmith-', suffix='.cl') as tmp:
        runtime, status, _, stderr = clsmith.clsmith('-o', tmp.name, *flags)

        # A non-zero exit status of clsmith implies that no program was
        # generated. Try again:
        if status:
            if depth > 100:
                logging.error(stderr)
                raise OSError("Failed to produce CLSmith program after 100 attempts")
            else:
                return _make_clsmith_program(session, *flags, depth=depth + 1)

        src = fs.read_file(tmp.name)

    # Check if the program is a duplicate. If so, try again:
    sha1 = crypto.sha1_str(src)
    is_dupe = session.query(Program.id)\
        .filter(Program.generator == CLSmith.generator_t,
                Program.sha1 == sha1).first()
    if is_dupe:
        return _make_clsmith_program(session, *flags, depth=depth + 1)

    program = Program(
        generator=CLSmith.generator_t,
        sha1=sha1,
        generation_time=runtime,
        linecount=len(src.split("\n")),
        charcount=len(src),
        src=src)
    session.add(program)
    session.commit()


class CLSmith(Generator):
    __name__ = "clsmith"
    generator_t = Generators.CLSMITH

    __harnesses__ = {
        None: Cl_launcher,
        "cl_launcher": Cl_launcher,
        "clang": Clang,
    }

    def num_programs(self, session: session_t=None) -> int:
        """ return the number of generated programs in the database """
        with ReuseSession(session) as s:
            return s.query(func.count(Program.id))\
                .filter(Program.generator == self.generator_t)\
                .scalar()

    def sloc_total(self, session: session_t=None) -> int:
        """ return the total linecount of generated programs """
        with ReuseSession(session) as s:
            return s.query(func.sum(Program.linecount))\
                .filter(Program.generator == self.generator_t)\
                .scalar()

    def generation_time(self, session: session_t=None) -> float:
        """ return the total generation time of all programs """
        with ReuseSession(session) as s:
            return s.query(func.sum(Program.generation_time))\
                .filter(Program.generator == self.generator_t)\
                .scalar()

    def num_testcases(self, session: session_t=None) -> int:
        """ return the total number of testcases """
        with ReuseSession(session) as s:
            return s.query(func.count(Testcase.id))\
                .join(Program)\
                .filter(Program.generator == self.generator_t)\
                .scalar()

    def generate(self, n: int=math.inf, up_to: int=math.inf) -> None:
        """ generate 'n' new programs 'up_to' this many exist in db """
        with Session(commit=False) as s:
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
                estimated_time = (self.generation_time(s) / num_progs) * num_to_generate
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
                _make_clsmith_program(s)

                # Update progress bar
                num_progs = self.num_programs(s)
                bar.update(num_progs)
                num_to_generate = max_value - num_progs
        print(f"All done! You now have {Colors.BOLD}{num_progs}{Colors.END} "
              "programs in the database")


class DSmith(Generator):
    __name__ = "dsmith"
    generator_t = Generators.DSMITH

    __harnesses__ = {
        None: Cldrive,
        "cldrive": Cldrive,
        "clang": Clang,
    }

    def num_programs(self, session: session_t=None) -> int:
        """ return the number of generated programs in the database """
        with ReuseSession(session) as s:
            return s.query(func.count(Program.id))\
                .filter(Program.generator == Generators.DSMITH)\
                .scalar()

    def sloc_total(self, session: session_t=None) -> int:
        """ return the total linecount of generated programs """
        with ReuseSession(session) as s:
            return s.query(func.sum(Program.linecount))\
                .filter(Program.generator == Generators.DSMITH)\
                .scalar()

    def num_testcases(self, session: session_t=None) -> int:
        """ return the total number of testcases """
        with ReuseSession(session) as s:
            return s.query(func.count(Testcase.id))\
                .join(Program)\
                .filter(Program.generator == self.generator_t)\
                .scalar()

    def generate(self, n: int=math.inf, up_to: int=math.inf) -> None:
        """ generate 'n' new programs 'up_to' this many exist in db """
        raise NotImplementedError
