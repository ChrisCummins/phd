# Copyright (C) 2015 Chris Cummins.
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

import sqlite3 as sql

import labm8 as lab
from labm8 import db
from labm8 import fs

class TestDatabase(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDatabase, self).__init__(*args, **kwargs)

        # Load test database
        self.db = db.Database("tests/data/db.sql")
        self.db_empty = db.Database("tests/data/db_empty.sql")

    # table_exists()
    def test_table_exists(self):
        self._test(True, self.db.table_exists("names"))
        self._test(True, self.db.table_exists("prices"))

    def test_table_exists_not(self):
        self._test(False, self.db.table_exists("not_a_real_table"))

    def test_table_exists_empty(self):
        self._test(False, self.db_empty.table_exists("names"))
        self._test(False, self.db_empty.table_exists("prices"))

    # get_tables()
    def test_get_tables(self):
        self._test(["names","prices"], self.db.get_tables())

    def test_get_tables_empty(self):
        self._test([], self.db_empty.get_tables())

    # drop_table(), create_table()
    def test_create_drop_tables(self):
        fs.rm("/tmp/labm8.sql")
        _db = db.Database("/tmp/labm8.sql")

        _db.drop_table("foo")
        self._test(False, _db.table_exists("foo"))
        _db.create_table("foo", (("id", "integer primary key"),))
        self._test(True, _db.table_exists("foo"))
        _db.drop_table("foo")
        self._test(False, _db.table_exists("foo"))


    # attach(), detach()
    def test_attach_detach(self):
        cmd = 'SELECT first from foo.names where first="Joe"'
        # Attach "db" to "db_empty" as "foo"
        self.db_empty.attach(self.db.path, "foo")
        # Run query on attached foo table.
        self._test(("Joe",), self.db_empty.execute(cmd).fetchone())
        # Detach "foo".
        self.db_empty.detach("foo")
        # Check that a query on "foo" raises error.
        self.assertRaises(sql.OperationalError, self.db_empty.execute, cmd)

    def test_multiple_attach(self):
        # Check that attaching a database twice raises an error.
        self.db_empty.attach(self.db.path, "foo")
        self.assertRaises(sql.OperationalError,
                          self.db_empty.attach, self.db.path, "foo")
        # Check that attaching the same database using a different name works.
        self.db_empty.attach(self.db.path, "bar")
        # Clean up.
        self.db_empty.detach("foo")
        self.db_empty.detach("bar")

    def test_multiple_detach(self):
        self.db_empty.attach(self.db.path, "foo")
        self.db_empty.detach("foo")
        # Check that detaching an already detached database raises an
        # error.
        self.assertRaises(sql.OperationalError,
                          self.db_empty.detach, "foo")

    def test_missing_detach(self):
        # Check that detaching an unknown database raises an error.
        self.assertRaises(sql.OperationalError,
                          self.db_empty.detach, "bar")

if __name__ == '__main__':
    main()
