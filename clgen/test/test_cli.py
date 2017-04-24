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
from unittest import TestCase, main, skip

import os

from clgen import cli


def get_parser():
    parser = cli.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("-c", action="store_true")
    return parser


class TestArgumentParser(TestCase):
    def test_args(self):
        parser = get_parser()
        args = parser.parse_args(["foo", "2"])
        self.assertEqual(args.a, "foo")
        self.assertEqual(args.b, "2")

    def test_version(self):
        parser = get_parser()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(["--version"])

    def test_verbose(self):
        parser = get_parser()
        args = parser.parse_args(["foo", "--verbose", "2"])
        self.assertEqual(args.verbose, True)

    def test_debug(self):
        parser = get_parser()
        args = parser.parse_args(["foo", "--debug", "2"])
        self.assertEqual(args.debug, True)


def mymethod(a, b):
    c = a / b
    print("{a} / {b} = {c}".format(**vars()))
    return c


class TestCli(TestCase):
    def test_main(self):
        self.assertEqual(cli.main(mymethod, 4, 2), 2)

    def test_main_exception_handler(self):
        os.environ["DEBUG"] = ""
        with self.assertRaises(SystemExit):
            cli.main(mymethod, 1, 0)

    def test_main_exception_debug(self):
        os.environ["DEBUG"] = "1"
        with self.assertRaises(ZeroDivisionError):
            cli.main(mymethod, 1, 0)


if __name__ == "__main__":
    main()
