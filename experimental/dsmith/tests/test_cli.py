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
import os

import pytest

from experimental.dsmith import cli


def mymethod(a, b):
  c = a // b
  print("{a} / {b} = {c}".format(**vars()))
  return c


# @test.XFail(reason="FIXME: cli.run() returning None")
def test_run():
  assert cli.run(mymethod, 4, 2) == 2


def test_run_exception_handler():
  # When DEBUG env variable set, exception is caught and system exits
  os.environ["DEBUG"] = ""
  with test.Raises(SystemExit):
    cli.run(mymethod, 1, 0)


def test_run_exception_debug():
  # When DEBUG env variable set, exception is not caught
  os.environ["DEBUG"] = "1"
  with test.Raises(ZeroDivisionError):
    cli.run(mymethod, 1, 0)


def test_cli_version():
  cli.main(["--version"])
