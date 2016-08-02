#
# dbutil - Smith sqlite3 database utilities.
#
import sqlite3


def table_exists(db, table_name):
    c = db.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='" +
              table_name + "'")
    res = c.fetchone()
    c.close()
    return res and res[0]


def is_github(db):
    return table_exists(db, 'Repositories')
