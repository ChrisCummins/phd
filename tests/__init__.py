from unittest import TestCase

import os

class TestDataNotFoundException(Exception): pass


def data_path(path, exists=True):
    """
    Return absolute path to unittest data file. Data files are located in
    tests/data.

    Args:
        path (str): Relative path.

    Returns:
        string: Absolute path.

    Raises:
        TestDataNotFoundException: If path doesn"t exist.
    """
    abspath = os.path.join(os.path.dirname(__file__), "data", path)
    if exists and not os.path.exists(abspath):
        raise TestDataNotFoundException(abspath)
    return abspath


def data_str(path):
    """
    Return contents of unittest data file as a string.

    Args:
        path (str): Relative path.

    Returns:
        string: File contents.

    Raises:
        TestDataNotFoundException: If path doesn't exist.
    """
    with open(data_path(path)) as infile:
        return infile.read()


def db_path(path):
    """
    Return absolute path to unittest data file. Data files are located in
    tests/data/db.

    Args:
        path (str): Relative path.

    Returns:
        string: Absolute path.

    Raises:
        TestDataNotFoundException: If path doesn't exist.
    """
    return data_path(os.path.join("db", str(path) + ".db"))


def db(name, **kwargs):
    """
    Return absolute path to unittest data file. Data files are located in
    tests/data/db.

    Args:
        path (str): Relative path.

    Returns:
        sqlite.Connection: Sqlite connection to database.

    Raises:
        TestDataNotFoundException: If path doesn't exist.
    """
    import sqlite3
    path = data_path(db_path(name), **kwargs)
    return sqlite3.connect(path)

class TestCLgen(TestCase):
    def test_hello_world(self):
        self.assertEqual(True, True)
