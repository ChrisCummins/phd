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

import pandas.io.sql as panda

import labm8 as lab
from labm8 import fs
from labm8 import io

if lab.is_python3():
    from io import StringIO
else:
    from StringIO import StringIO


class Error(Exception):
    """
    Module-level base error class.
    """
    pass


class SchemaError(Error):
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

    @property
    def tables(self):
        """
        Returns a list of table names.

        Example:

            >>> db.tables
            ["bar", "foo"]

        Returns:

            list of str: One string for each table name.
        """
        select = ("SELECT name FROM sqlite_master",)
        query = self.execute(*select)
        result = query.fetchall()

        # Filter first column from rows.
        return [row[0] for row in result]

    @property
    def schema(self):
        """
        Returns the schema of all tables.

        For each table, return the name, and a list of tuples
        representing the columns. Each column tuple consists of a
        (name, type) pair. Note that additional metadata, such as
        whether a column may be null, or whether a column is a primary
        key, is not returned.

        Example:

            >>> db.schema
            [("bar", (("id", "integer"), ("name", "table"))]

        Returns:

            list of tuples: Each tuple has the format (name, columns), where
              "columns" is a list of tuples of the form (name, type).
        """
        def _info2columns(info):
            return tuple((column["name"], column["type"]) for column in info)

        def _table2tuple(table):
            return (table, _info2columns(self.table_info(table)))

        return [_table2tuple(table) for table in self.tables]

    def table_info(self, table):
        """
        Returns information about the named table.

        See: https://www.sqlite.org/pragma.html#pragma_table_info

        Example:

            >>> db.table_info("foo")
            [{"name": "id", "type": "integer", "primary key": True,
              "notnull": False, "default_value": None}]

        Arguments:

            name (str): The name of the table to lookup.

        Returns:

            list of dicts: One dict per column. Each dict contains the
              keys: "name", "type", "primary key", "notnull", and
              "default_value".

        Raises:

            sql.OperationalError: If table does not exist.
        """
        def _row2dict(row):
            return {
                "name": row[1],
                "type": row[2].lower(),
                "notnull": row[3] == 1,
                "default_value": row[4],
                "primary_key": row[5] == 1
            }

        if table not in self.tables:
            raise sql.OperationalError("Cannot retrieve information about "
                                       "missing table '{0}'".format(table))

        query = self.execute("PRAGMA table_info({table})".format(table=table))
        return [_row2dict(row) for row in query]

    def isempty(self, tables=None):
        """
        Return whether a table or the entire database is empty.

        A database is empty is if it has no tables. A table is empty
        if it has no rows.

        Arguments:

            tables (sequence of str, optional): If provided, check
              that the named tables are empty. If not provided, check
              that all tables are empty.

        Returns:

            bool: True if tables are empty, else false.

        Raises:

            sql.OperationalError: If one or more of the tables do not
              exist.
        """
        tables = tables or self.tables

        for table in tables:
            if self.num_rows(table) > 0:
                return False

        return True

    def num_rows(self, table):
        """
        Return the number of rows in the named table.

        Example:

            >>> db.num_rows("foo")
            3

        Arguments:

            table (str): The name of the table to count the rows in.

        Returns:

            int: The number of rows in the named table.

        Raises:

            sql.OperationalError: If the named table does not exist.
        """
        return self.execute("SELECT Count(*) from " + table).fetchone()[0]

    def close(self):
        """
        Close a database connection.
        """
        self.connection.close()

    def drop_table(self, name):
        """
        Drop an existing table.

        If the table does not exist, nothing happens.
        """
        if name in self.tables:
            self.execute("DROP TABLE " + name)

    def create_table(self, name, schema):
        """
        Create a new table.

        If the table already exists, nothing happens.

        Example:

            >>> db.create_table("foo", (("id", "integer primary key"),
                                        ("value", "text")))

        Arguments:

           name (str): The name of the table to create.
           schema (sequence of tuples): A list of (name, type) tuples
             representing each of the columns.
        """
        columns = [" ".join(column) for column in schema]
        self.execute("CREATE TABLE IF NOT EXISTS {name} ({columns})"
                     .format(name=name, columns=",".join(columns)))

    def create_table_from(self, name, src):
        """
        Create a new table with same schema as the source.

        If the named table already exists, nothing happens.

        Arguments:

            name (str): The name of the table to create.
            src (str): The name of the source table to duplicate.

        Raises:

            sql.OperationalError: If source table does not exist.
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
        new_cmd = re.sub("(CREATE TABLE) \w+", "\\1 " + name, cmd,
                         re.IGNORECASE)

        # Execute this new command.
        self.execute(new_cmd)

    def copy_table(self, src, dst):
        """
        Create a carbon copy of the source table.

        Arguments:

            src (str): The name of the table to copy.
            dst (str): The name of the target duplicate table.

        Raises:

            sql.OperationalError: If source table does not exist.
        """
        # Create table.
        self.create_table_from(dst, src)

        # Copy contents of src to dst.
        self.execute("INSERT INTO {dst} SELECT * FROM {src}"
                     .format(dst=dst, src=src))

        # Commit changes.
        self.commit()

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

    def export_csv(self, table, output=None, **kwargs):
        """
        Export a table to a CSV file.

        If an output path is provided, write to file. Else, return a
        string.

        Wrapper around pandas.sql.to_csv(). See:
        http://pandas.pydata.org/pandas-docs/stable/io.html#io-store-in-csv

        Arguments:

            table (str): Name of the table to export.
            output (str, optional): Path of the file to write.
            **kwargs: Additional args passed to pandas.sql.to_csv()

        Returns:

            str: CSV string, or None if writing to file.

        Raises:

            IOError: In case of error writing to file.
            SchemaError: If the named table is not found.
        """
        # Determine if we're writing to a file or returning a string.
        isfile = output is not None
        output = output or StringIO()

        if table not in self.tables:
            raise SchemaError("Cannot find table '{table}'"
                              .format(table=table))

        # Don't print row indexes by default.
        if "index" not in kwargs:
            kwargs["index"] = False

        table = panda.read_frame("select * from {table}".format(table=table),
                                 self.connection)
        table.to_csv(output, **kwargs)
        return None if isfile else output.getvalue()

    def arff_attr(self, table, types=None):
        """
        Export a table schema to a list of arff attributes.

        Arff types can be either NUMERIC or NOMINAL. If the arff types
        are not provided, the types will be derived from the SQL
        schema specification for the table. INTEGER and REAL types
        will be converted to NUMERIC, and all else will be converted
        to NOMINAL. NOMINAL values are a set of all unique values.

        Example:

            >>> db.arff_attr("names")
            (("id", "NUMERIC"), ("first", ("David","Joe")),
             ("last", ("Bloggs","Bowie","Brent")))

        Arguments:

            table (str): Name of the table to export.
            types (sequence of str, optional): The arff attribute
              types to use.

        Returns:

            list of tuples: Where each tuple is a (name,type) pair
              describing an SQL column. The "type" is itself a tuple
              of str, either a single str containing a names type, or
              a set of nominal category values.

        Raises:

            SchemaError: If the named table is not found.
        """
        def _sql_type2arff_type(ctype):
            if ctype == "integer":
                return "NUMERIC"
            if ctype == "real":
                return "NUMERIC"
            return "NOMINAL"

        def _columns2types(columns):
            return [_sql_type2arff_type(column["type"]) for column in columns]

        def _expand_nominal(cname):
            query = self.execute("SELECT DISTINCT {col} FROM {table}"
                                 .format(col=cname, table=table))
            distinct = sorted(set(row[0] for row in query))

            return cname, tuple(distinct)

        def _expand_nominals(attr):
            return tuple([_expand_nominal(cname) if ctype == "NOMINAL"
                          else (cname, (ctype,)) for cname, ctype in attr])

        if table not in self.tables:
            raise SchemaError("Cannot find table '{table}'"
                              .format(table=table))

        info = self.table_info(table)
        names = [column["name"] for column in info]
        types = types or _columns2types(info)

        return _expand_nominals(zip(names, types))

    def export_arff(self, table, output=None, types=None, relation=None):
        """
        Export a table to Weka arff format.

        If an output path is provided, write to file. Else, return a
        string. If the arff types are not provided, the types will be
        derived from the SQL schema specification for the
        table. INTEGER and REAL types will be converted to NUMERIC,
        and all else will be converted to nominal specification.

        Arguments:

            table (str): Name of the table to export.
            output (str, optional): Path of the file to write.
            types (sequence of str, optional): The arff attribute
              types to use.
            relation (str, optional): The name of the relation. If
              not given, defaults to the name of the table.

        Returns:

            str: ARFF string, or None if writing to file.

        Raises:

            IOError: In case of error writing to file.
            SchemaError: If the named table is not found.
        """
        # Determine if we're writing to a file or returning a string
        # and create either StringIO or file object.
        isfile = output is not None
        if isfile:
            if lab.is_python3():
                output = open(output, "w", newline="")
            else:
                output = open(output, "wb")
        else:
            output = StringIO()

        attributes = self.arff_attr(table, types=types)
        relation = relation or table
        writer = csv.writer(output)

        # Write header.
        writer.writerow(("@RELATION " + relation,))
        writer.writerow([])

        # Write schema.
        for attribute in attributes:
            aname, atype = attribute

            if len(atype) > 1:
                # Write out all attribute values.
                sname, stype = StringIO(), StringIO()
                namewriter, typewriter = csv.writer(sname), csv.writer(stype)
                namewriter.writerow((aname,))
                typewriter.writerow(atype)
                output.write("@ATTRIBUTE {name} {{{type}}}\n"
                             .format(name=sname.getvalue().rstrip(),
                                     type=stype.getvalue().rstrip()))
            else:
                writer.writerow(("@ATTRIBUTE " + aname, atype[0]))

        # Write body.
        writer.writerow([])
        writer.writerow(["@DATA"])
        query = self.execute("SELECT * FROM {table}".format(table=table))
        writer.writerows(query)

        if isfile:
            output.close()
        else:
            return output.getvalue()
