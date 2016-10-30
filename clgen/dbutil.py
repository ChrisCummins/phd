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

import clgen


class DatabaseException(clgen.CLgenError):
    pass


def create_db(path, github=False):
    path = os.path.expanduser(path)

    if os.path.exists(path):
        raise DatabaseException("Database '{}' already exists"
                                .format(path))

    print("creating database ...".format(path))
    db = sqlite3.connect(path)
    c = db.cursor()
    if github:
        script = clgen.sql_script('create-gh-samples-db')
    else:
        script = clgen.sql_script('create-samples-db')
    c.executescript(script)
    c.close()
    db.commit()


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
    db = sqlite3.connect(path)
    c = db.cursor()
    c.execute('SELECT Count(*) FROM PreprocessedFiles WHERE status=0')
    return c.fetchone()[0]
