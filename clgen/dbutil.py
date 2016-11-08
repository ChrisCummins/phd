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
"""
CLgen sqlite3 database utilities
"""
from __future__ import absolute_import, division, print_function

import os
import sqlite3

from hashlib import md5

import clgen


def create_db(path, github=False):
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


class md5sum_aggregator:
    def __init__(self):
        self.md5 = md5()

    def step(self, value):
        self.md5.update(str(value).encode('utf-8'))

    def finalize(self):
        return self.md5.hexdigest()


class linecount_aggregator:
    def __init__(self):
        self.count = 0

    def step(self, value):
        self.count += len(value.split('\n'))

    def finalize(self):
        return self.count


class charcount_aggregator:
    def __init__(self):
        self.count = 0

    def step(self, value):
        self.count += len(value)

    def finalize(self):
        return self.count


def connect(db_path):
    """
    Returns a connection to a database.

    Database has additional aggregate functions:

        MD5SUM() returns md5 of column values
        LC() returns sum line count of text columns
        CC() returns sum character count of text columns

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


def is_modified(db):
    c = db.cursor()

    c.execute("SELECT value FROM Meta WHERE key='preprocessed_checksum'")
    result = c.fetchone()
    cached_checksum = result[0] if result else None

    c.execute('SELECT MD5SUM(id) FROM ContentFiles')
    checksum = c.fetchone()[0]
    c.close()

    return False if cached_checksum == checksum else checksum


def set_modified_status(db, checksum):
    c = db.cursor()
    c.execute("INSERT OR REPLACE INTO Meta VALUES (?,?)",
              ('preprocessed_checksum', checksum))
    db.commit()
    c.close()


def table_exists(db, table_name):
    c = db.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='" +
              table_name + "'")
    res = c.fetchone()
    c.close()
    return res and res[0]


def is_github(db):
    return table_exists(db, 'Repositories')


def num_good_kernels(path):
    db = connect(path)
    c = db.cursor()
    c.execute('SELECT Count(*) FROM PreprocessedFiles WHERE status=0')
    return c.fetchone()[0]


def remove_preprocessed(path):
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
