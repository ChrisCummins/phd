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
import humanize
import math
import progressbar
from sqlalchemy.sql import func

import dsmith
import dsmith.opencl.db

from dsmith import Colors
from dsmith.langs import Driver, Generator, Language
from dsmith.opencl.db import *


class Cldrive(Driver):
    __name__ = "cldrive"


class Cl_launcher(Driver):
    __name__ = "cl_launcher"


class CLSmith(Generator):
    __name__ = "clsmith"
    generator_t = Generators.CLSMITH

    __drivers__ = {
        None: Cl_launcher,
        "cl_launcher": Cl_launcher,
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

    def generate(self, n: int=math.inf, up_to: int=math.inf) -> None:
        """ generate 'n' new programs 'up_to' this many exist in db """

        with Session(commit=False) as s:
            num_progs = self.num_programs(s)

            if n == math.inf and up_to == math.inf:
                max_value = progressbar.UnknownLength
            elif n == math.inf:
                max_value = up_to
            else:
                max_value = num_progs + n

            if num_progs >= max_value:
                print(f"There are already {Colors.BOLD}{num_progs}{Colors.END} "
                      "programs in the database. Nothing to be done.")

            num_to_generate = max_value - num_progs
            estimated_time = (self.generation_time(s) / num_progs) * num_to_generate
            eta = humanize.naturaldelta(datetime.timedelta(seconds=estimated_time))
            print(f"{Colors.BOLD}{num_to_generate}{Colors.END} programs are "
                  "to be generated. Estimated generation time: " +
                  f"{Colors.BOLD}{eta}{Colors.END}.")

            bar = progressbar.ProgressBar(initial_value=num_progs,
                                          max_value=max_value)

            print(f"CLSMITH GENERATE {num_progs} {max_value}")



class DSmith(Generator):
    __name__ = "dsmith"

    __drivers__ = {
        None: Cldrive,
        "cldrive": Cldrive,
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

    def generate(self, n: int=math.inf, up_to: int=math.inf) -> None:
        """ generate 'n' new programs 'up_to' this many exist in db """
        raise NotImplementedError


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
        generator = self.__generators__.get(name)
        if not generator:
            raise ValueError("Unknown generator")
        return generator()
