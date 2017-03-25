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
"""
CLgen sqlite3 database utilities
"""
import os
import re
import sqlite3

from hashlib import md5

import clgen


def create_db(path: str, github: bool=False) -> None:
    """
    Create an empty OpenCL kernel database.

    Arguments:
        path (str): Path to database to create.
        github (bool, optional): Add tables for GitHub metadata.
    """
    path = os.path.expanduser(path)

    if os.path.exists(path):
        raise clgen.UserError("'{}' already exists".format(path))

    db = sqlite3.connect(path)
    c = db.cursor()
    if github:
        script = clgen.sql_script('create-gh-samples-db')
    else:
        script = clgen.sql_script('create-samples-db')
    c.executescript(script)
    c.close()
    db.commit()
    db.close()


class md5sum_aggregator:
    """
    sqlite3 aggregator for computing checksum of column values.
    """
    def __init__(self):
        self.md5 = md5()

    def step(self, value) -> None:
        self.md5.update(str(value).encode('utf-8'))

    def finalize(self) -> str:
        return self.md5.hexdigest()


class linecount_aggregator:
    """
    sqlite3 aggregator for computing line count of column values.
    """
    def __init__(self):
        self.count = 0

    def step(self, value) -> None:
        self.count += len(value.split('\n'))

    def finalize(self) -> int:
        return self.count


class charcount_aggregator:
    """
    sqlite3 aggregator for computing character count of column values.
    """
    def __init__(self):
        self.count = 0

    def step(self, value) -> None:
        self.count += len(value)

    def finalize(self) -> int:
        return self.count


def connect(db_path: str):
    """
    Returns a connection to a database.

    Database has additional aggregate functions:

     * MD5SUM() returns md5 of column values
     * LC() returns sum line count of text columns
     * CC() returns sum character count of text columns

    Arguments:

        db_path (str): Path to database

    Returns:

        sqlite3 connection
    """
    db = sqlite3.connect(db_path)
    db.create_aggregate("MD5SUM", 1, md5sum_aggregator)
    db.create_aggregate("LC", 1, linecount_aggregator)
    db.create_aggregate("CC", 1, charcount_aggregator)
    return db


def set_meta(path: str, key: str, value: str) -> None:
    """
    Set a value in a database's Meta table.

    If the a row with this key already exists in the table, it is replaced.

    Arguments:
        path (str): Path to database.
        key (str): Key name.
        value (str): Value to insert.
    """
    db = sqlite3.connect(path)
    c = db.cursor()
    c.execute("DELETE FROM Meta WHERE key=?", (key,))
    c.execute("INSERT INTO Meta (key, value) VALUES (?,?)",
              (key, value))
    c.close()
    db.commit()


def get_meta(path: str, key: str) -> str:
    """
    Retrieve a value from a database's Meta table.

    Arguments:
        path (str): Path to database.
        key (str): Key name.

    Returns:
        str: Value. If no row matching key is found, returns empty string.
    """
    db = sqlite3.connect(path)
    c = db.cursor()
    c.execute("SELECT value FROM Meta WHERE key=?", (key,))
    v = c.fetchone()
    if v:
        return v[0]
    else:
        return ""


def set_version_meta(path: str, version: str=clgen.version()) -> None:
    """
    Set the "version" key in an database.

    This is useful for marking version requirements of specific datasets, e.g.
    a databse schema which requires a particular CLgen version, or a scheme
    which is likely to change in the future.

    Arguments:
        path (str): Path to database.
        version (str, optional): Version value (defaults to CLgen version).
    """
    set_meta(path, "version", version)


def version_meta_matches(path: str, version: str=clgen.version()) -> bool:
    """
    Check that the "version" key in a database matches the expected value.

    If the database does not have a "version" key in the Meta table, returns
    False.

    Arguments:
        path (str): Path to database.
        version (str, optional): Version value (defaults to CLgen version).

    Returns:
        bool: True if version in database matches expected version, else False.
    """
    return get_meta(path, "version") == version


def run_script(path: str, script: str) -> None:
    """
    Run an SQL script on a databse.

    Arguments:
        path (str): Path to database.
        script (str): Name of SQL data script.
    """
    db = sqlite3.connect(path)
    c = db.cursor()
    c.executescript(clgen.sql_script(script))
    c.close()
    db.commit()
    db.close()


def is_modified(db) -> bool:
    """
    Returns whether database is preprocessed.

    Arguments:
        db (sqlite3.Connection): Database.

    Returns:
        bool: True if database is modified, else False.
    """
    c = db.cursor()

    c.execute("SELECT value FROM Meta WHERE key='preprocessed_checksum'")
    result = c.fetchone()
    cached_checksum = result[0] if result else None

    c.execute('SELECT MD5SUM(id) FROM ContentFiles')
    checksum = c.fetchone()[0]
    c.close()

    return False if cached_checksum == checksum else checksum


def set_modified_status(db, checksum: str) -> None:
    """
    Set database preprocessed checksum.

    Arguments:
        db (sqlite3.Connection): Database.
        checksum (str): New preprocessed checksum.
    """
    c = db.cursor()
    c.execute("INSERT OR REPLACE INTO Meta VALUES (?,?)",
              ('preprocessed_checksum', checksum))
    db.commit()
    c.close()


def table_exists(db, table_name: str) -> None:
    """
    SQL table exists.

    Arguments:
        db (sqlite3.Connection): Database.
        table_name (str): Name of table.

    Returns:
        bool: True if table with name exists.
    """
    c = db.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='" +
              table_name + "'")
    res = c.fetchone()
    c.close()
    return res and res[0]


def is_github(db) -> None:
    """
    SQL table has GitHub metadata tables.

    Arguments:
        db (sqlite3.Connection): Database.

    Returns:
        bool: True if GitHub tables in database.
    """
    return table_exists(db, 'Repositories')


def num_good_kernels(path: str) -> int:
    """
    Fetch the number of good preprocessed kernels from dataset.

    Arguments:

        path (str): Path to database.

    Returns:

        int: Num preprocessed kernels where status is 0.
    """
    return num_rows_in(path, "PreprocessedFiles", "WHERE status=0")


def num_rows_in(path: str, table: str, condition: str="") -> int:
    """
    Fetch number of rows in table.

    Arguments:

        path (str): Path to database.
        table (str): Table ID.
        condition (str, optional): Conditional.

    Returns:

        int: Num rows.
    """
    db = connect(path)
    c = db.cursor()
    c.execute('SELECT Count(*) FROM {table} {condition}'
              .format(table=table, condition=condition))
    return c.fetchone()[0]


def cc(path: str, table: str, column: str="Contents", condition: str="") -> int:
    """
    Fetch character count of contents in table.

    Arguments:

        path (str): Path to database.
        table (str): Table ID.
        condition (str, optional): Conditional.

    Returns:

        int: Num lines.
    """
    db = connect(path)
    c = db.cursor()
    c.execute("SELECT CC({column}) FROM {table} {condition}"
              .format(column=column, table=table, condition=condition))
    return c.fetchone()[0] or 0


def lc(path: str, table: str, column: str="Contents", condition: str="") -> int:
    """
    Fetch line count of contents in table.

    Arguments:

        path (str): Path to database.
        table (str): Table ID.
        condition (str, optional): Conditional.

    Returns:

        int: Num lines.
    """
    db = connect(path)
    c = db.cursor()
    c.execute("SELECT LC({column}) FROM {table} {condition}"
              .format(column=column, table=table, condition=condition))
    return c.fetchone()[0] or 0


def remove_preprocessed(path: str) -> None:
    """
    Removes all preprocessed files from database.

    ContentFiles remain unchanged.

    Arguments:

        path (str): Path to database.
    """
    db = connect(path)
    c = db.cursor()
    c.execute("DELETE FROM PreprocessedFiles")
    c.execute("DELETE FROM Meta WHERE key='preprocessed_checksum'")
    c.close()
    db.commit()


def remove_bad_preprocessed(db_path: str) -> None:
    """
    Remove all ugly and bad contents from PreprocessedFiles table.

    Arguments:
        db_path (str): Dataset.
    """
    original_size = fs.du(db_path, human_readable=False)
    original_size_human_readable = fs.du(db_path, human_readable=True)
    log.info("vacuuming", original_size_human_readable, "database")
    sys.stdout.flush()

    # Remove contents from bad or ugly preprocessed files.
    db = dbutil.connect(db_path)
    c = db.cursor()
    c.execute("UPDATE PreprocessedFiles SET contents='[DELETED]' "
              "WHERE status=1 OR status=2")
    db.commit()
    c.close()

    c = db.cursor()
    c.execute("VACUUM")
    db.commit()
    c.close()

    new_size = fs.du(db_path, human_readable=False)
    new_size_human_readable = fs.du(db_path, human_readable=True)
    reduction_ratio = (1 - (new_size / original_size)) * 100
    log.info("done. new size {}. ({:.0f}% reduction)"
             .format(new_size_human_readable, reduction_ratio), sep=".")


def sql_insert_dict(c, table: str, data: dict) -> None:
    """
    Insert a dict of key value pairs into an SQL table.

    Uses the key names as column names, as the values as column values.

    Arguments:
        c (sqlite3.Cursor): Database cursor.
        table (str): Destination table.
        data (dict): Key value pairs.
    """
    cmd = ("INSERT INTO {table}({cols}) VALUES({vals})"
           .format(table=table,
                   cols=','.join(data.keys()),
                   vals=','.join(['?'] * len(data))))

    c.execute(cmd, tuple(data.values()))


_sql_rm_chars = re.compile(r'[\(\)]')
_sql_sub_chars = re.compile(r'-')


def escape_sql_key(key: str) -> str:
    """
    Escape SQL key.

    Arguments:
        key (str): SQL key.

    Returns:
        str: Escaped key.
    """
    return re.sub(_sql_sub_chars, '_',
                  re.sub(_sql_rm_chars, '', '_'.join(key.split(' '))))
