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

import datetime

class TestTime(TestCase):

    def test_strfmt(self):
        date = datetime.date(1970, 1, 1)
        self._test("1970-01-01 00:00:00", lab.time.strfmt(date))

    def test_now(self):
        self._test("datetime", type(lab.time.now()).__name__)

    def test_nowstr(self):
        self._test("str", type(lab.time.nowstr()).__name__)
