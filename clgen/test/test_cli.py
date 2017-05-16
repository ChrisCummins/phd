#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
import pytest
import os

from clgen import cli


def _mymethod(a, b):
    c = a // b
    print("{a} / {b} = {c}".format(**vars()))
    return c


@pytest.mark.xfail(reason="FIXME: cli.run() returning None")
def test_run():
    assert cli.run(_mymethod, 4, 2) == 2


def test_run_exception_handler():
    os.environ["DEBUG"] = ""
    with pytest.raises(SystemExit):
        cli.run(_mymethod, 1, 0)


def test_run_exception_debug():
    os.environ["DEBUG"] = "1"
    with pytest.raises(ZeroDivisionError):
        cli.run(_mymethod, 1, 0)
