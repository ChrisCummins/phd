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
import os
import pytest

from labm8 import fs

from dsmith import test as tests
from dsmith import cli


def mymethod(a, b):
    c = a // b
    print("{a} / {b} = {c}".format(**vars()))
    return c


# @pytest.mark.xfail(reason="FIXME: cli.run() returning None")
def test_run():
    assert cli.run(mymethod, 4, 2) == 2


def test_run_exception_handler():
    # When DEBUG env variable set, exception is caught and system exits
    os.environ["DEBUG"] = ""
    with pytest.raises(SystemExit):
        cli.run(mymethod, 1, 0)


def test_run_exception_debug():
    # When DEBUG env variable set, exception is not caught
    os.environ["DEBUG"] = "1"
    with pytest.raises(ZeroDivisionError):
        cli.run(mymethod, 1, 0)


def test_cli_version():
    with pytest.raises(SystemExit):
        cli.main("--version")


def test_cli_test_cache_path():
    with pytest.raises(SystemExit):
        cli.main("test --cache-path".split())


def test_cli_test_coverage_path():
    with pytest.raises(SystemExit):
        cli.main("test --coverage-path".split())


def test_cli_test_coveragerc_path():
    with pytest.raises(SystemExit):
        cli.main("test --coveragerc-path".split())
