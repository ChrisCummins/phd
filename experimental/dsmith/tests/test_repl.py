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
import pytest

from experimental.dsmith import repl


def test_execute_type_error():
  with test.Raises(TypeError):
    repl._execute(2.5)


def test_execute_unrecognized():
  with test.Raises(ValueError):
    repl._execute("__ unrecognized input ___")

  with test.Raises(repl.UnrecognizedInput):
    repl._execute("__ unrecognized input ___")


def test_run_command_type_error():
  with test.Raises(TypeError):
    repl.run_command(2.5)
