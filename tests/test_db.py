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

        # Make a copy of test databases.
        fs.cp("tests/data/db.sql", "/tmp/labm8.db.sql")
        fs.cp("tests/data/db_empty.sql", "/tmp/labm8.db_empty.sql")

        # Load test databases.
        self.db = db.Database("/tmp/labm8.db.sql")
        self.db_empty = db.Database("/tmp/labm8.db_empty.sql")

    # tables attribute
    def test_tables(self):
        self._test(["names","prices"], self.db.tables)

    def test_tables_setter(self):
        with self.assertRaises(AttributeError) as ctx:
            self.db.tables = ["foo", "bar"]

    def test_tables_empty(self):
        self._test([], self.db_empty.tables)

    def test_table_exists(self):
        self._test(True, "names" in self.db.tables)
        self._test(True, "prices" in self.db.tables)

    def test_table_exists_not(self):
        self._test(False, "not_a_real_table" in self.db.tables)

    def test_table_exists_empty(self):
        self._test(False, "names" in self.db_empty.tables)
        self._test(False, "prices" in self.db_empty.tables)

    # table_info()
    def test_table_info(self):
        self._test([{
            "name": "id",
            "type": "integer",
            "primary_key": True,
            "notnull": False,
            "default_value": None
        }, {
            "name": "description",
            "type": "text",
            "primary_key": False,
            "notnull": False,
            "default_value": None
        }, {
            "name": "price",
            "type": "real",
            "primary_key": False,
            "notnull": False,
            "default_value": None
        }], self.db.table_info("prices"))

    def test_table_info_missing_table(self):
        # table_info() raises an error if table does not exist.
        with self.assertRaises(sql.OperationalError) as ctx:
            self.db.table_info("foo")

    # schema attribute
    def test_schema(self):
        self._test([("names",
                     (("first",       "text"),
                      ("last",        "text"))),
                    ("prices",
                     (("id",          "integer"),
                      ("description", "text"),
                      ("price",       "real")))],
                   self.db.schema)

    def test_schema_empty(self):
        self._test([], self.db_empty.schema)

    # num_rows()
    def test_num_rows(self):
        self._test(3, self.db.num_rows("names"))
        self._test(0, self.db.num_rows("prices"))

    def test_num_rows_missing_table(self):
        # Error is raised if table does not exist.
        with self.assertRaises(sql.OperationalError) as ctx:
            self.db.num_rows("foo")

    # isempty()
    def test_isempty(self):
        self._test(True, self.db_empty.isempty())
        self._test(False, self.db.isempty())
        self._test(True, self.db.isempty(("prices",)))

    def test_isempty_missing_table(self):
        # Error is raised if table does not exist.
        with self.assertRaises(sql.OperationalError) as ctx:
            self.db.isempty("foo")

    # drop_table(), create_table()
    def test_create_drop_tables(self):
        fs.rm("/tmp/labm8.sql")
        _db = db.Database("/tmp/labm8.sql")

        _db.drop_table("foo")
        self._test(False, "foo" in _db.tables)
        _db.create_table("foo", (("id", "integer primary key"),))
        self._test(True, "foo" in _db.tables)
        _db.drop_table("foo")
        self._test(False, "foo" in _db.tables)

    # Constructor "schema" argument
    def test_constructor_schema(self):
        fs.rm("/tmp/labm8.sql")
        _db = db.Database("/tmp/labm8.sql", {
            "foo": (("id", "integer"), ("prop", "text"))
        })
        self._test(["foo"], _db.tables)

    # copy_table()
    def test_copy_table(self):
        cmd = 'SELECT first from names_cpy where first="Joe"'

        self.db.drop_table("names_cpy")
        self.assertRaises(sql.OperationalError, self.db.execute, cmd)

        # Copy table "names" to "names_cpy".
        self.db.copy_table("names", "names_cpy")
        # Check that there's a "names_cpy" table.
        self._test(True, "names_cpy" in self.db.tables)
        # Run query on copied table.
        self._test(("Joe",), self.db.execute(cmd).fetchone())
        # Drop copied table.
        self.db.drop_table("names_cpy")

    def test_copy_table_no_src(self):
        # Copying a non-existent table raises an error.
        self.assertRaises(sql.OperationalError,
                          self.db.copy_table, "foo", "foo_cpy")

    def test_copy_table_dst_exists(self):
        self.db.drop_table("names_cpy")

        # Copying the same table twice raises an error.
        self.db.copy_table("names", "names_cpy")
        self.assertRaises(sql.OperationalError,
                          self.db.copy_table, "names", "names_cpy")
        # Drop copied table.
        self.db.drop_table("names_cpy")


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
