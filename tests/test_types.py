# Copyright (C) 2015, 2016 Chris Cummins.
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

import labm8
from labm8 import types


class TestTypes(TestCase):
    def test_is_str(self):
        self.assertTrue(types.is_str("Hello, World!"))
        self.assertTrue(types.is_str(str("Hello, World!")))

        if labm8.is_python3():
            self.assertFalse(types.is_str("Hello, World!".encode("utf-8")))
            self.assertFalse(types.is_str(bytes("Hello, World!".encode("utf-8"))))
        else:
            self.assertTrue(types.is_str("Hello, World!".encode("utf-8")))
            self.assertTrue(types.is_str(bytes("Hello, World!".encode("utf-8"))))

        self.assertFalse(types.is_str(8))
        self.assertFalse(types.is_str(['a', 'b', 'c']))
        self.assertFalse(types.is_str({'a': 'b', 'c': 18}))
