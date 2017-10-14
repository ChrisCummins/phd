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

from typing import List, Tuple


class Harness(object):
    """
    Abstract interface for a test harness.

    Attributes:
        process_result_t (Tuple[float, int, str, str]): Process result.
        __name__ (str): Harness name.
    """
    def __repr__(self):
        return self.__name__

    def run(self, testbed, testcase) -> None:
        """ execute a testcase and record the result """
        raise NotImplementedError("abstract class")


class Generator(object):
    """
    Abstract interface for a program generator.

    Attributes:
        __name__ (str): Generator name.
        __harnesses__ (List[Generator]): List of available harnesses.
    """
    def __repr__(self):
        return self.__name__

    def num_programs(self) -> int:
        """ return the number of generated programs in the database """
        raise NotImplementedError("abstract class")

    def sloc_total(self) -> int:
        """ return the total linecount of generated programs """
        raise NotImplementedError("abstract class")

    def generation_time(self) -> float:
        """ return the total generation time of all programs """
        raise NotImplementedError("abstract class")

    def num_testcases(self) -> int:
        """ return the total number of testcases """
        raise NotImplementedError("abstract class")

    def generate(self, n: int=math.inf, up_to: int=math.inf) -> None:
        """ generate 'n' new programs, until 'up_to' exist in db """
        raise NotImplementedError("abstract class")

    @property
    def harnesses(self):
        names = (name for name in self.__harnesses__ if name)
        return (self.__harnesses__[name]() for name in names)

    def mkharness(self, name: str) -> Harness:
        """ Instantiate harness from string """
        harness = self.__harnesses__.get(name)
        if not harness:
            raise ValueError(f"Unknown {self.__name__} harness '{name}'")
        return harness()


class Language(object):
    """
    Abstract interface for a programming language.

    Attributes:
        __name__ (str): Language name.
        __generators__ (List[Generator]): List of available generators.
    """
    def __repr__(self):
        return self.__name__

    def mktestcases(self, generator: Generator=None) -> None:
        """ Generate testcases, optionally for a specific generator """
        raise NotImplementedError("abstract class")

    @property
    def generators(self):
        names = (name for name in self.__generators__ if name)
        return (self.__generators__[name]() for name in names)

    def mkgenerator(self, name: str) -> Generator:
        """ Instantiate generator from string """
        generator = self.__generators__.get(name)
        if not generator:
            raise ValueError(f"Unknown {self.__name__} generator '{generator}'")
        return generator()

    @property
    def testbeds(self) -> List['Testbed']:
        """ Return all testbeds in data store """
        raise NotImplementedError("abstract class")

    @property
    def available_testbeds(self) -> List['Testbed']:
        """ Return all testbeds on the current machine """
        raise NotImplementedError("abstract class")

    def run_testcases(self, testbeds: List['Testbed'],
                      pairs: List[Tuple[Generator, Harness]]) -> None:
        """ Run testcases """
        raise NotImplementedError("abstract class")


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
        raise ValueError(f"Unknown language '{name}'")
    return lang()
