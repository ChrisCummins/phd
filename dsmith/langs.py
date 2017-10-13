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
Programming language.

Attributes:
    __languages__ (Dict[str, Language]): List of all available languages.
"""
import math

from collections import namedtuple


class Driver(object):
    """
    Abstract interface for a driver.

    Attributes:
        process_result_t (Tuple[float, int, str, str]): Process result.
        __name__ (str): Driver name.
    """
    process_result_t = namedtuple('result_t', ['runtime', 'returncode', 'stdout', 'stderr'])

    def drive(self, testcase, **params) -> process_result_t:
        """ drive a testcase """
        raise NotImplementedError("abstract class")


class Generator(object):
    """
    Abstract interface for a program generator.

    Attributes:
        __name__ (str): Generator name.
        __drivers__ (List[Generator]): List of available drivers.
    """

    def num_programs(self) -> int:
        """ return the number of generated programs in the database """
        raise NotImplementedError("abstract class")

    def sloc_total(self) -> int:
        """ return the total linecount of generated programs """
        raise NotImplementedError("abstract class")

    def generation_time(self, session: session_t=None) -> float:
        """ return the total generation time of all programs """
        raise NotImplementedError("abstract class")

    def generate(self, n: int=math.inf, up_to: int=math.inf) -> None:
        """ generate 'n' new programs, until 'up_to' exist in db """
        raise NotImplementedError("abstract class")


class Language(object):
    """
    Abstract interface for a programming language.

    Attributes:
        __name__ (str): Language name.
        __generators__ (List[Generator]): List of available generators.
    """
    def mkgenerator(self, name: str) -> Generator:
        raise NotImplementedError("abstract class")

    @property
    def generators(self):
        names = (name for name in self.__generators__ if name)
        return (self.__generators__[name]() for name in names)



# Deferred importing of languages, since the modules may need to import this
# file.
from dsmith.opencl import OpenCL

__languages__ = {
    "opencl": OpenCL,
}


def mklang(name: str) -> Language:
    """
    Instantiate language from name.
    """
    lang = __languages__.get(name)
    if not lang:
        raise ValueError("Unknown language")
    return lang()
