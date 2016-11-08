#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
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
from unittest import TestCase
import tests

import os

from labm8 import fs

import clgen
from clgen import dbutil


class TestDbutil(TestCase):
    def test_table_exists(self):
        self.assertTrue(dbutil.table_exists(
            tests.db('empty'), 'ContentFiles'))
        self.assertFalse(dbutil.table_exists(
            tests.db('empty'), 'NotATable'))

    def test_is_github(self):
        self.assertFalse(dbutil.is_github(tests.db('empty')))
        self.assertTrue(dbutil.is_github(tests.db('empty-gh')))

    def test_remove_preprocessed(self):
        tmpdb = 'test_remove_preprocessed.db'
        fs.cp(tests.db_path('10-kernels-preprocessed'), tmpdb)

        self.assertEqual(8, dbutil.num_good_kernels(tmpdb))
        db = dbutil.connect(tmpdb)
        self.assertFalse(dbutil.is_modified(db))
        db.close()

        dbutil.remove_preprocessed(tmpdb)

        self.assertEqual(0, dbutil.num_good_kernels(tmpdb))

        db = dbutil.connect(tmpdb)
        self.assertTrue(dbutil.is_modified(db))
        db.close()

        fs.rm(tmpdb)


if __name__ == '__main__':
    main()
