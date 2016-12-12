#
# dbutil - Smith sqlite3 database utilities.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sqlite3

import smith


class DatabaseException(smith.SmithException): pass


def create_db(path, github=False):
    path = os.path.expanduser(path)

    if os.path.exists(path):
        raise DatabaseException("Database '{}' already exists"
                                .format(path))

    print("creating database ...".format(path))
    db = sqlite3.connect(path)
    c = db.cursor()
    if github:
        script = smith.sql_script('create-gh-samples-db')
    else:
        script = smith.sql_script('create-samples-db')
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
