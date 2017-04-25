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


def _get_parser():
    parser = cli.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("-c", action="store_true")
    return parser


def test_args():
    parser = _get_parser()
    args = parser.parse_args(["foo", "2"])
    assert args.a == "foo"
    assert args.b == "2"


def test_version():
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])


def test_verbose():
    parser = _get_parser()
    args = parser.parse_args(["foo", "--verbose", "2"])
    assert args.verbose


def test_debug():
    parser = _get_parser()
    args = parser.parse_args(["foo", "--debug", "2"])
    assert args.debug


# CLI


def _mymethod(a, b):
    c = a / b
    print("{a} / {b} = {c}".format(**vars()))
    return c


def test_main():
    assert cli.main(_mymethod, 4, 2) == 2


def test_main_exception_handler():
    os.environ["DEBUG"] = ""
    with pytest.raises(SystemExit):
        cli.main(_mymethod, 1, 0)


def test_main_exception_debug():
    os.environ["DEBUG"] = "1"
    with pytest.raises(ZeroDivisionError):
        cli.main(_mymethod, 1, 0)
