# Copyright (C) 2015-2017 Chris Cummins.
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

from labm8 import fs
from labm8 import make

class TestMake(TestCase):

    # make()
    def test_make(self):
        ret, out, err = make.make(dir="tests/data/makeproj")
        self._test(0, ret)
        self._test(True, out is not None)
        self._test(True, fs.isfile("tests/data/makeproj/foo"))
        self._test(True, fs.isfile("tests/data/makeproj/foo.o"))

    def test_make_bad_target(self):
        with self.assertRaises(make.NoTargetError):
            make.make(target="bad-target", dir="tests/data/makeproj")

    def test_make_bad_target(self):
        with self.assertRaises(make.NoMakefileError):
            make.make(dir="/bad/path")
        with self.assertRaises(make.NoMakefileError):
            make.make(target="foo", dir="tests/data")

    def test_make_fail(self):
        with self.assertRaises(make.MakeError):
            make.make(target="fail", dir="tests/data/makeproj")

    # clean()
    def test_make_clean(self,):
        fs.cd("tests/data/makeproj")
        make.make()
        self._test(True, fs.isfile("foo"))
        self._test(True, fs.isfile("foo.o"))
        make.clean()
        self._test(False, fs.isfile("foo"))
        self._test(False, fs.isfile("foo.o"))
        fs.cdpop()
