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

import sys

import labm8 as lab


class TestLabm8(TestCase):

    # exit()
    def test_exit(self):
        with self.assertRaises(SystemExit) as ctx:
            lab.exit(0)
        self.assertEqual(ctx.exception.code, 0)
        with self.assertRaises(SystemExit) as ctx:
            lab.exit(1)
        self.assertEqual(ctx.exception.code, 1)

    # is_python3()
    def test_is_python3(self):
        if sys.version_info >= (3, 0):
            self._test(True, lab.is_python3())
        else:
            self._test(False, lab.is_python3())
