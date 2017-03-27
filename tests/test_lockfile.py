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

import os
import re

import labm8 as lab
from labm8 import fs
from labm8 import lockfile

class TestLockFile(TestCase):

    def test_path(self):
        lock = lockfile.LockFile("/tmp/labm8.lock")
        fs.rm(lock.path)

        self.assertFalse(lock.islocked)

        lock.acquire()
        self.assertTrue(fs.exists(lock.path))
        self.assertTrue(lock.islocked)
        self.assertTrue(lock.owned_by_self)

        lock.acquire()
        self.assertTrue(lock.islocked)
        self.assertTrue(lock.owned_by_self)

        lock.release()
        self.assertFalse(fs.exists(lock.path))
