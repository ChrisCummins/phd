# Copyright (C) 2015 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
from unittest import TestCase, main

import labm8 as lab
import labm8.fs

import os

class TestFs(TestCase):

    # path()
    def test_path(self):
        expected = os.path.abspath(".") + "/foo/bar"
        actual = lab.fs.path("foo", "bar")
        print(actual)
        self.assertTrue(actual == expected)

    def test_path_relpath(self):
        expected = "foo/bar"
        actual = lab.fs.path("foo", "bar", abspath=False)
        print(actual)
        self.assertTrue(actual == expected)

    # is_subdir()
    def test_is_subdir(self):
        expected = True
        actual = lab.fs.is_subdir("/home", "/")
        print(actual)
        self.assertTrue(actual == expected)

        expected = True
        actual = lab.fs.is_subdir("/proc/1", "/proc")
        print(actual)
        self.assertTrue(actual == expected)

    def test_is_subdir_same(self):
        expected = True
        actual = lab.fs.is_subdir("/proc/1", "/proc/1/")
        print(actual)
        self.assertTrue(actual == expected)

    def test_is_subdir_not_subdir(self):
        expected = False
        actual = lab.fs.is_subdir("/", "/home")
        print(actual)
        self.assertTrue(actual == expected)

    # pwd()
    def test_pwd(self):
        expected = os.getcwd()
        actual = lab.fs.pwd()
        print(actual)
        self.assertTrue(actual == expected)

    # read()
    def test_read_empty(self):
        expected = []
        actual = lab.fs.read("tests/data/empty_file")
        print(actual)
        self.assertTrue(actual == expected)

    def test_read_hello_world(self):
        expected = ['Hello, world!']
        actual = lab.fs.read("tests/data/hello_world")
        print(actual)
        self.assertTrue(actual == expected)

    def test_read_data1(self):
        expected = [
            '# data1 - test file',
            'This',
            'is a test file',
            'With',
            'trailing  # comment',
            '',
            '',
            '',
            'whitespace',
            '0.344'
        ]
        actual = lab.fs.read("tests/data/data1")
        print(actual)
        self.assertTrue(actual == expected)

    def test_read_data1_comment(self):
        expected = [
            'This',
            'is a test file',
            'With',
            'trailing',
            '',
            '',
            '',
            'whitespace',
            '0.344'
        ]
        actual = lab.fs.read("tests/data/data1", comment_char="#")
        print(actual)
        self.assertTrue(actual == expected)

    def test_read_data1_no_rstrip(self):
        expected = [
            '# data1 - test file\n',
            'This\n',
            'is a test file\n',
            'With\n',
            'trailing  # comment  \n',
            '\n',
            '\n',
            '\n',
            'whitespace\n',
            '0.344\n'
        ]
        actual = lab.fs.read("tests/data/data1", rstrip=False)
        print(actual)
        self.assertTrue(actual == expected)

if __name__ == '__main__':
    main()
