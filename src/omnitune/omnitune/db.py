import atexit
import sqlite3 as sql

import labm8
from labm8 import fs
from labm8 import io


class Database(object):
    def __init__(self, path):
        """
        Arguments:
            path The path to the database file.
            schema A list of tuple pairs, of the form: (name, type).
        """
        self.path = fs.path(path)
        self.connection = sql.connect(self.path)
        self.cursor = self.connection.cursor()
        self.tables = {}

        io.debug("Opened connection to '{0}'".format(self.path))

        # Register exit handler
        atexit.register(self.close)

    def close(self):
        self.connection.close()
        io.debug("Closed connection to '{0}'".format(self.path))

    def create_table(self, name, schema):
        self.tables[name] = schema

        schema_str = "(" + ", ".join([" ".join([str(a) for a in x])
                                     for x in schema]) + ")"
        cmd_str = "CREATE TABLE IF NOT EXISTS " + name + "\n" + schema_str
        self.cursor.execute(cmd_str)
        # io.debug(cmd_str)

    def escape_value(self, table, i, value):
        if self.tables[table][i][1].upper() == "TEXT":
            return "'" + str(value) + "'"
        else:
            return value

    def _insert(self, table, values, ignore_duplicates=False):
        escaped_values = [self.escape_value(table, i, values[i])
                          for i in range(len(values))]

        cmd = ["INSERT"]
        if ignore_duplicates:
            cmd.append("OR IGNORE")
        cmd.append("INTO")
        cmd.append(table)
        cmd.append("VALUES (")
        cmd.append(", ".join([str(x) for x in escaped_values]))
        cmd.append(")")

        cmd_str = " ".join(cmd)
        # io.debug(cmd_str)

        self.cursor.execute(cmd_str)
        self.connection.commit()

    def insert(self, table, values):
        return self._insert(table, values)

    def insert_unique(self, table, values):
        return self._insert(table, values, ignore_duplicates=True)

    def execute(self, *args):
        return self.cursor.execute(*args)

    def commit(self):
        return self.connection.commit()

    def select(self, table, select, where):
        cmd = ["SELECT", select, "FROM", table, "WHERE", where]
        cmd_str = " ".join(cmd)
        return self.cursor.execute(cmd_str)

    def select1(self, table, select, where):
        return self.select(self, table, select, where).fetchone()

    def count(self, table, where):
        cmd = ["SELECT Count(*) FROM", table, "WHERE", where]
        cmd_str = " ".join(cmd)
        return self.cursor.execute(cmd_str).fetchone()[0]
