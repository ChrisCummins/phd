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

import labm8 as lab
from labm8 import types


class TestTypes(TestCase):
    # is_str()
    def test_is_str(self):
        self.assertTrue(types.is_str("Hello, World!"))
        self.assertTrue(types.is_str(str("Hello, World!")))

        if lab.is_python3():
            self.assertFalse(types.is_str("Hello, World!".encode("utf-8")))
            self.assertFalse(types.is_str(bytes("Hello, World!".encode("utf-8"))))
        else:
            self.assertTrue(types.is_str("Hello, World!".encode("utf-8")))
            self.assertTrue(types.is_str(bytes("Hello, World!".encode("utf-8"))))

        self.assertFalse(types.is_str(8))
        self.assertFalse(types.is_str(['a', 'b', 'c']))
        self.assertFalse(types.is_str({'a': 'b', 'c': 18}))

    def test_is_str_seq(self):
        self._test(False, types.is_str(tuple([1])))
        self._test(False, types.is_str((1, 2)))
        self._test(False, types.is_str([1]))
        self._test(False, types.is_str([1, 2]))

    def test_is_str_num(self):
        self._test(False, types.is_str(1))
        self._test(False, types.is_str(1.3))

    def test_is_str_dict(self):
        self._test(False, types.is_str({"foo": 100}))
        self._test(False, types.is_str({10: ["a", "b", "c"]}))


    # is_dict() tests
    def test_is_dict(self):
        self._test(True, types.is_dict({"foo": 100}))
        self._test(True, types.is_dict({10: ["a", "b", "c"]}))

    def test_is_dict_str(self):
        self._test(False, types.is_dict("a"))
        self._test(False, types.is_dict("abc"))
        self._test(False, types.is_dict(["abc", "def"][0]))

    def test_is_dict_seq(self):
        self._test(False, types.is_dict(tuple([1])))
        self._test(False, types.is_dict((1, 2)))
        self._test(False, types.is_dict([1]))
        self._test(False, types.is_dict([1, 2]))

    def test_is_dict_num(self):
        self._test(False, types.is_dict(1))
        self._test(False, types.is_dict(1.3))


    # is_seq() tests
    def test_is_seq(self):
        self._test(True, types.is_seq(tuple([1])))
        self._test(True, types.is_seq((1, 2)))
        self._test(True, types.is_seq([1]))
        self._test(True, types.is_seq([1, 2]))

    def test_is_seq_str(self):
        self._test(False, types.is_seq("a"))
        self._test(False, types.is_seq("abc"))
        self._test(False, types.is_seq(["abc", "def"][0]))

    def test_is_seq_num(self):
        self._test(False, types.is_seq(1))
        self._test(False, types.is_seq(1.3))

    def test_is_seq_dict(self):
        self._test(False, types.is_seq({"foo": 100}))
        self._test(False, types.is_seq({10: ["a", "b", "c"]}))


    # flatten()
    def test_flatten(self):
        self._test([1, 2, 3], types.flatten([[1], [2, 3]]))
