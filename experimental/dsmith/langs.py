#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
import sys
from pathlib import Path
from typing import Iterable
from typing import Tuple

from experimental import dsmith
from experimental.dsmith import Colors


class Harness(dsmith.ReprComparable):
  """
  Abstract interface for a test harness.

  Attributes:
      process_result_t (Tuple[float, int, str, str]): Process result.
      __name__ (str): Harness name.
      __generators__ (List[Generator]): List of compatible generators.
  """

  # Abstract methods (must be implemented):

  def run_testcases(self, testbeds: Iterable["Testbed"]) -> None:
    """ execute testcases on the specified testbeds and record the results """
    raise NotImplementedError("abstract class")

  def testbeds(self) -> Iterable["Testbed"]:
    """ return all testbeds in the data store """
    pass

  def available_testbeds(self) -> Iterable["Testbed"]:
    """ return testbeds available on this machine """
    pass

  # Default methods (may be overriden):

  def __repr__(self):
    return f"{Colors.BOLD}{Colors.YELLOW}{self.__name__}{Colors.END}"

  @property
  def generators(self):
    names = (name for name in self.__generators__ if name)
    return (self.__generators__[name]() for name in names)


class Generator(dsmith.ReprComparable):
  """
  Abstract interface for a program generator.

  Attributes:
      __name__ (str): Generator name.
      __harnesses__ (List[Generator]): List of compatible harnesses.
  """

  # Abstract methods (must be implemented):

  # TODO: refactor these away
  # def num_programs(self) -> int:
  #     """ return the number of generated programs in the database """
  #     raise NotImplementedError("abstract class")

  # def sloc_total(self) -> int:
  #     """ return the total linecount of generated programs """
  #     raise NotImplementedError("abstract class")

  # def generation_time(self) -> float:
  #     """ return the total generation time of all programs """
  #     raise NotImplementedError("abstract class")

  # def num_testcases(self) -> int:
  #     """ return the total number of testcases """
  #     raise NotImplementedError("abstract class")

  def generate(self, n: int = math.inf, up_to: int = math.inf) -> None:
    """ generate 'n' new programs, until 'up_to' exist in db """
    raise NotImplementedError("abstract class")

  def import_from_dir(self, indir: Path) -> None:
    """ import program sources from a directory """
    raise NotImplementedError("abstract class")

  # Default methods (may be overriden):

  def __repr__(self):
    return f"{Colors.BOLD}{Colors.GREEN}{self.__name__}{Colors.END}"

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


class Language(dsmith.ReprComparable):
  """
  Abstract interface for a programming language.

  Attributes:
      __name__ (str): Language name.
      __generators__ (Iterable[Generator]): List of available generators.
      __harnesses__ (Iterable[Harness]): List of available harnesses.
  """

  # Abstract methods (must be implemented):

  def describe_testbeds(
    self, available_only: bool = False, file=sys.stdout
  ) -> None:
    """ describe testbeds """
    raise NotImplementedError("abstract class")

  # def describe_results(self, file=sys.stdout) -> None:
  #     """ describe results """
  #     raise NotImplementedError("abstract class")

  # def mktestcases(self, generator: Generator=None) -> None:
  #     """ Generate testcases, optionally for a specific generator """
  #     raise NotImplementedError("abstract class")

  # def testbeds(self) -> List['Testbed']:
  #     """ Return all testbeds in data store """
  #     raise NotImplementedError("abstract class")

  # def available_testbeds(self) -> List['Testbed']:
  #     """ Return all testbeds on the current machine """
  #     raise NotImplementedError("abstract class")

  # def run_testcases(self, testbeds: List['Testbed'],
  #                   pairs: List[Tuple[Generator, Harness]]) -> None:
  #     """ Run testcases """
  #     raise NotImplementedError("abstract class")

  # Default methods (may be overriden):

  def __repr__(self):
    return f"{Colors.BOLD}{Colors.RED}{self.__name__}{Colors.END}"

  @property
  def generators(self):
    names = (name for name in self.__generators__ if name)
    return (self.__generators__[name]() for name in names)

  @property
  def harnesses(self):
    names = (name for name in self.__harnesses__ if name)
    return (self.__harnesses__[name]() for name in names)

  def mkgenerator(self, name: str) -> Generator:
    """ Instantiate generator from string """
    generator = self.__generators__.get(name)
    if not generator:
      raise ValueError(f"Unknown {self} generator '{name}'")
    return generator()

  def mkharness(self, name: str) -> Generator:
    """ Instantiate harness from string """
    harness = self.__harnesses__.get(name)
    if not harness:
      raise ValueError(f"Unknown {self} harness '{name}'")
    return harness()


def mklang(name: str) -> Language:
  """
  Instantiate language from name.

  Raises:
      LookupError: If language is not found.
  """
  # Global table of available languages
  langs = dict()

  if dsmith.WITH_OPENCL:
    # Deferred importing of languages to break circular dependencies from
    # language modules which require this file.
    from dsmith.opencl import OpenCL

    langs["opencl"] = OpenCL

  if dsmith.WITH_SOLIDITY:
    from dsmith.sol import Solidity

    langs["solidity"] = Solidity
    langs["sol"] = Solidity

  if dsmith.WITH_GLSL:
    from dsmith.glsl import Glsl

    langs["glsl"] = Glsl

  lang = langs.get(name)
  if not lang:
    raise LookupError(f"Unknown language '{name}'")
  return lang()
