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
The OpenCL programming language.
"""
import sys

from experimental.dsmith.langs import Generator, Harness, Language
from experimental.dsmith.opencl import db
from experimental.dsmith.opencl import difftest
from experimental.dsmith.opencl.db import *
from experimental.dsmith.opencl.generators import CLSmith, DSmith, RandChar, \
  RandTok
from experimental.dsmith.opencl.harnesses import Cl_launcher, Clang, Cldrive


class OpenCL(Language):
  __name__ = "opencl"

  __generators__ = {
      None: DSmith,  # default
      "dsmith": DSmith,
      "clsmith": CLSmith,
      "randchar": RandChar,
      "randtok": RandTok,
  }

  __harnesses__ = {
      "cldrive": Cldrive,
      "cl_launcher": Cl_launcher,
      "clang": Clang,
  }

  def __init__(self):
    if db.engine is None:
      db.init()

  def mktestbeds(self, string: str) -> List[Testbed]:
    """ Instantiate testbed(s) by name """
    with Session() as s:
      return [
          TestbedProxy(testbed)
          for testbed in Testbed.from_str(string, session=s)
      ]

  def run_testcases(self, testbeds: List[str],
                    pairs: List[Tuple[Generator, Harness]]) -> None:
    with Session() as s:
      for generator, harness in pairs:
        for testbed_name in testbeds:
          testbed = Testbed.from_str(testbed_name, session=s)[0]
          self._run_testcases(testbed, generator, harness, s)

  def describe_testbeds(self, available_only: bool = False,
                        file=sys.stdout) -> None:
    with Session() as s:
      if not available_only:
        print(
            f"The following {self} testbeds are in the data store:", file=file)
        for harness in sorted(self.harnesses):
          for testbed in sorted(harness.testbeds()):
            print(
                f"    {harness} {testbed} {testbed.platform} on {testbed.host}",
                file=file)
        print(file=file)

      print(
          f"The following {self} testbeds are available on this machine:",
          file=file)
      for harness in sorted(self.harnesses):
        for testbed in sorted(harness.available_testbeds()):
          print(f"    {harness} {testbed} {testbed.platform}", file=file)

  def describe_results(self, file=sys.stdout) -> None:
    with Session() as s:
      for harness in self.harnesses:
        for generator in harness.generators:
          for testbed in harness.testbeds():
            testbed = str(testbed)
            num_results = harness.num_results(generator, testbed)
            if num_results:
              word_num = humanize.Commas(num_results)
              print(
                  f"There are {Colors.BOLD}{word_num}{Colors.END} "
                  f"{generator}:{harness} "
                  f"results on {testbed}.",
                  file=file)

  def difftest(self) -> None:
    difftest.difftest()
