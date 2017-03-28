#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
import tests

from clgen import fetch


class TestFetch(TestCase):
    def test_inline_fs_headers(self):
        src = fetch.inline_fs_headers(tests.data_path("cl", "sample-3.cl"))
        self.assertTrue(src.contains("MY_DATA_TYPE"))
        self.assertTrue(src.contains("__kernel void"))


if __name__ == "__main__":
    main()
