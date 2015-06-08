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
import atexit
import csv
import re
import six
import sqlite3 as sql

import labm8 as lab
from labm8 import fs
from labm8 import io


class Error(Exception):
    """
    Module-level base error class.
    """
    pass


class SchemaError(Exception):
    """
    Error thrown in case of conflicting schemas.
    """
    pass


class Database(object):
    def __init__(self, path, tables={}):
        """
        Arguments:
            path (str): The path to the database file.
            tables (dictionary of {str: tuple of str}, optional): A diction
              of {name: schema} pairs, where a schema is list of tuple pairs,
              of the form: (name, type).
        """
        self.path = fs.path(path)

        # Create directory if needed.
        parent_dir = fs.dirname(path)
        if parent_dir:
            fs.mkdir(parent_dir)

        self.connection = sql.connect(self.path)

        for name,schema in six.iteritems(tables):
            self.create_table(name, schema)

        io.debug("Opened connection to '{0}'".format(self.path))

        # Register exit handler
        atexit.register(self.close)

    def table_exists(self, name):
        """
        Check if a table exists.

        Arguments:

            name (str): The name of the table to check whether it exists

        Returns:

            bool: True if table exists, else False.
        """
        select = ("SELECT name FROM sqlite_master WHERE name = ?", (name,))
        query = self.execute(*select)
        result = query.fetchone()

        return result is not None and len(result) == 1

    def get_tables(self):
        """
        Returns a list of table names.

        Example:

            >>> db.get_tables()
            ["bar", "foo"]

        Returns:

            list of str: One string for each table.
        """
        select = ("SELECT name FROM sqlite_master",)
        query = self.execute(*select)
        result = query.fetchall()

        # Filter first column from rows.
        return [row[0] for row in result]

    def isempty(self):
        """
        Return whether the database is empty.

        A database is empty is if it has no tables.

        Returns:

            bool: True if database is empty, else false.
        """
        return len(self.get_tables()) == 0

    def close(self):
        """
        Close a database connection.
        """
        self.connection.close()
        io.debug("Closed connection to '{0}'".format(self.path))

    def drop_table(self, name):
        """
        Drop an existing table.

        If the table does not exist, nothing happens.
        """
        if self.table_exists(name):
            self.execute("DROP TABLE " + name)

    def create_table(self, name, schema):
        """
        Create a new table.

        If the table already exists, nothing happens.
        """
        constraints = [" ".join(constraint) for constraint in schema]
        cmd = [
            "CREATE TABLE IF NOT EXISTS ",
            name,
            " (", ",".join(constraints),
            ")"
        ]
        self.execute("".join(cmd))

    def copy_table(self, src, dst):
        """
        Copy a table schema, and all of its contents.

        Arguments:

            src (str): The name of the table to copy.
            dst (str): The name of the target duplicate table.
        """
        # Lookup the command which was used to create the "src" table.
        query = self.execute("SELECT sql FROM sqlite_master WHERE "
                             "type='table' and name=?", (src,))
        try:
            cmd = query.fetchone()[0]
        except TypeError:
            raise sql.OperationalError("Cannot copy non-existent table '{0}'"
                                       .format(src))

        # Modify the original command to replace the old table name
        # with the new one.
        new_cmd = re.sub("(CREATE TABLE) \w+", "\\1 " + dst, cmd, re.IGNORECASE)

        # Execute this new command.
        self.execute(new_cmd)

        # Copy contents of src to dst.
        self.execute("INSERT INTO {dst} SELECT * FROM {src}"
                     .format(dst=dst, src=src))

    def execute(self, *args):
        """
        Execute the given arguments.
        """
        return self.connection.cursor().execute(*args)

    def commit(self):
        """
        Commit the current transaction.

        Make sure to call this method after you've modified the
        database's state!
        """
        return self.connection.commit()

    def attach(self, path, name):
        """
        Attach a database.

        Arguments:

            path (str): Path to the database to merge.
            name (str): Name to attach database as.
        """
        self.execute("ATTACH ? as ?", (path, name))

    def detach(self, name):
        """
        Detach a database.

        Arguments:

            name (str): Name of database to detach.
        """
        self.execute("DETACH ?", (name,))

    def merge(self, rhs):
        """
        Merge the contents of the supplied database.

        Arguments:
            rhs Another Database instance to merge into this database.

        Raises:
            SchemaError If the schema of the merged database does not match.
        """
        # Throw an "eppy" if the schemas do not match.
        if self.get_tables() != rhs.get_tables():
            raise SchemaError("Schema of merged table does not match")

        self.attach(rhs.path, "rhs")

        for table in self.get_tables():
            self.execute("INSERT OR IGNORE INTO {0} SELECT * FROM rhs.{0}"
                         .format(table))

        # Tidy up.
        self.commit()
        self.detach("rhs")
