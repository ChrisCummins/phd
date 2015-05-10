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
from unittest import main
from tests import TestCase

import labm8 as lab
import labm8.fs

import os

class TestFs(TestCase):

    # path()
    def test_path(self):
        self._test(os.path.abspath(".") + "/foo/bar",
                   lab.fs.path("foo", "bar"))

    def test_path_relpath(self):
        self._test("foo/bar",
                   lab.fs.path("foo", "bar", abspath=False))

    # is_subdir()
    def test_is_subdir(self):
        self._test(True, lab.fs.is_subdir("/home", "/"))
        self._test(True, lab.fs.is_subdir("/proc/1", "/proc"))
        self._test(True, lab.fs.is_subdir("/proc/1", "/proc/1/"))

    def test_is_subdir_not_subdir(self):
        self._test(False,
                   lab.fs.is_subdir("/", "/home"))

    # pwd()
    def test_pwd(self):
        self._test(os.getcwd(), lab.fs.pwd())

    # read()
    def test_read_empty(self):
        self._test([],
                   lab.fs.read("tests/data/empty_file"))

    def test_read_hello_world(self):
        self._test(['Hello, world!'],
                   lab.fs.read("tests/data/hello_world"))

    def test_read_data1(self):
        self._test([
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
        ],
                   lab.fs.read("tests/data/data1"))

    def test_read_data1_comment(self):
        self._test([
            'This',
            'is a test file',
            'With',
            'trailing',
            '',
            '',
            '',
            'whitespace',
            '0.344'
        ],
                   lab.fs.read("tests/data/data1", comment_char="#"))

    def test_read_data1_no_rstrip(self):
        self._test([
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
        ],
                   lab.fs.read("tests/data/data1", rstrip=False))

if __name__ == '__main__':
    main()
