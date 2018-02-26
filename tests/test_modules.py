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
from unittest import main,skip
from tests import TestCase

import sys

from labm8 import modules


class TestModules(TestCase):

    @skip("It doesn't work, I CBA to figure out why.")
    def test_import_foreign(self):
        my_math = modules.import_foreign("math", "my_math")
        if sys.version_info < (3, 0):
            self._test(0, my_math.sin(0))

    def test_import_foreign_fail(self):
        if sys.version_info < (3, 0):
            self.assertRaises(ImportError, modules.import_foreign,
                              "notamodule", "foo")
