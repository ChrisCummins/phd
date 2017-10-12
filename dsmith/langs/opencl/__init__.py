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
from sqlalchemy.sql import func

import dsmith
import dsmith.langs.opencl.db

from dsmith.langs import Driver, Generator, Language
from dsmith.langs.opencl.db import *


class Cldrive(Driver):
    __name__ = "cldrive"


class Cl_launcher(Driver):
    __name__ = "cl_launcher"


class CLSmith(Generator):
    __name__ = "clsmith"

    __drivers__ = {
        None: Cl_launcher,
        "cl_launcher": Cl_launcher,
    }

    @property
    def num_programs(self, session: session_t=None) -> int:
        """ return the number of generated programs in the database """
        with ReuseSession(session) as s:
            return s.query(Program)\
                .filter(Program.generator == Generators.CLSMITH)\
                .count()

    @property
    def sloc_total(self, session: session_t=None) -> int:
        """ return the total linecount of generated programs """
        with ReuseSession(session) as s:
            return s.query(func.sum(Program.linecount))\
                .filter(Program.generator == Generators.CLSMITH)\
                .scalar()


class DSmith(Generator):
    __name__ = "dsmith"

    __drivers__ = {
        None: Cldrive,
        "cldrive": Cldrive,
    }

    @property
    def num_programs(self, session: session_t=None) -> int:
        """ return the number of generated programs in the database """
        with ReuseSession(session) as s:
            return s.query(Program)\
                .filter(Program.generator == Generators.DSMITH)\
                .count()

    @property
    def sloc_total(self, session: session_t=None) -> int:
        """ return the total linecount of generated programs """
        with ReuseSession(session) as s:
            return s.query(func.sum(Program.linecount))\
                .filter(Program.generator == Generators.DSMITH)\
                .scalar()



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
