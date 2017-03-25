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
from unittest import TestCase
import tests

import os
import sqlite3

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

    def test_num_rows_in(self):
        self.assertEqual(10, dbutil.num_rows_in(tests.db_path('10-kernels'),
                                                "ContentFiles"))

        self.assertEqual(0, dbutil.num_rows_in(tests.db_path('10-kernels'),
                                               "PreprocessedFiles"))

        self.assertEqual(8, dbutil.num_rows_in(tests.db_path('10-kernels-preprocessed'),
                                               "PreprocessedFiles",
                                               "WHERE status=0"))

        self.assertEqual(2, dbutil.num_rows_in(tests.db_path('10-kernels-preprocessed'),
                                               "PreprocessedFiles",
                                               "WHERE status!=0"))

    def test_lc(self):
        self.assertEqual(1368, dbutil.lc(tests.db_path('10-kernels'),
                                         "ContentFiles"))

        self.assertEqual(
            0, dbutil.lc(tests.db_path('10-kernels'), "PreprocessedFiles"))

        self.assertEqual(
            865,
            dbutil.lc(tests.db_path('10-kernels-preprocessed'),
                      "PreprocessedFiles", condition="WHERE status=0"))

        self.assertEqual(
            2,
            dbutil.lc(tests.db_path('10-kernels-preprocessed'),
                      "PreprocessedFiles", condition="WHERE status!=0"))

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

    def test_remove_bad_preprocessed(self):
        fs.rm("tmp.db")
        dbutil.create_db("tmp.db")
        db = sqlite3.connect("tmp.db")
        c = db.cursor()

        # Create some data to test with:
        c.execute("DELETE FROM PreprocessedFiles")
        c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                  ("id1", 0, "good output"))
        c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                  ("id2", 1, "bad output"))
        c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                  ("id3", 2, "ugly output"))
        db.commit()
        c.close()

        # Check that data was written properly:
        c = db.cursor()
        c.execute("SELECT Count(*) FROM PreprocessedFiles")
        count = c.fetchone()[0]
        self.assertEqual(3, count)
        db.close()

        dbutil.remove_bad_preprocessed("tmp.db")

        # Check that clean worked:
        db = sqlite3.connect("tmp.db")
        c = db.cursor()
        c.execute("SELECT Count(*) FROM PreprocessedFiles")
        count = c.fetchone()[0]
        self.assertEqual(3, count)
        c.execute("SELECT contents FROM PreprocessedFiles WHERE status=1 "
                  "OR status=2")
        rows = c.fetchall()
        print(rows)
        self.assertTrue(all(not r == "[DELETED]" for r in rows))

        # Clean up:
        c.execute("DELETE FROM PreprocessedFiles")
        db.commit()
        c.close()

        # Check that clean-up worked:
        c = db.cursor()
        c.execute("SELECT Count(*) FROM PreprocessedFiles")
        count = c.fetchone()[0]
        self.assertEqual(0, count)
        fs.rm("tmp.db")


if __name__ == '__main__':
    main()
