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
from __future__ import division

from tests import TestCase

from labm8 import text

class TestText(TestCase):

    # get_substring_idxs()
    def test_get_substring_idxs(self):
        self.assertEqual([0, 2], text.get_substring_idxs('a', 'aba'))
        self.assertEqual([], text.get_substring_idxs('a', 'bb'))

    # truncate()
    def test_truncate(self):
        self._test("foo", text.truncate("foo", 100))
        self._test("1234567890", text.truncate("1234567890", 10))
        self._test("12345...", text.truncate("1234567890", 8))
        for i in range(10, 20):
            self._test(i,
                       len(text.truncate("The quick brown fox jumped "
                                         "over the slow lazy dog", i)))

    def test_truncate_bad_maxchar(self):
        with self.assertRaises(text.TruncateError):
            text.truncate("foo", -1)
            text.truncate("foo", 3)


    # distance()
    def test_levenshtein(self):
        self._test(0, text.levenshtein("foo", "foo"))
        self._test(1, text.levenshtein("foo", "fooo"))
        self._test(3, text.levenshtein("foo", ""))
        self._test(1, text.levenshtein("1234", "1 34"))
        self._test(1, text.levenshtein("123", "1 3"))

    # diff()
    def test_diff(self):
        self._test(0, text.diff("foo", "foo"))
        self._test(0.25, text.diff("foo", "fooo"))
        self._test(1, text.diff("foo", ""))
        self._test(0.25, text.diff("1234", "1 34"))
        self._test((1/3), text.diff("123", "1 3"))
