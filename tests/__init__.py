from __future__ import absolute_import, print_function, with_statement

import os
import sqlite3

from unittest import TestCase

import clgen

class TestData404(Exception): pass


def data_path(path, exists=True):
    """
    Return absolute path to unittest data file. Data files are located in
    tests/data.

    Args:
        path (str): Relative path.

    Returns:
        string: Absolute path.

    Raises:
        TestData404: If path doesn"t exist.
    """
    abspath = os.path.join(os.path.dirname(__file__), "data", path)
    if exists and not os.path.exists(abspath):
        raise TestData404(abspath)
    return abspath


def data_str(path):
    """
    Return contents of unittest data file as a string.

    Args:
        path (str): Relative path.

    Returns:
        string: File contents.

    Raises:
        TestData404: If path doesn't exist.
    """
    with open(data_path(path)) as infile:
        return infile.read()


def unpack_archive(path, compression="bz2"):
    """
    Unpack a compressed archive.

    Arguments:
        path (str): Path to archive.
        compression (str, optional): Archive compression type.
    """
    import tarfile
    tar = tarfile.open(path, "r:" + compression)
    tar.extractall()
    tar.close()


def archive(path):
    """
    Returns a text archive, unpacking if necessary.

    Arguments:
        path (str): Path to archive.

    Returns:
        str: Path to archive.
    """
    path = data_path(path)

    if not os.path.exists(archive):
        unpack_archive(path + ".tar.bz2")
    return path


def db_path(path):
    """
    Return absolute path to unittest data file. Data files are located in
    tests/data/db.

    Args:
        path (str): Relative path.

    Returns:
        string: Absolute path.

    Raises:
        TestData404: If path doesn't exist.
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
        TestData404: If path doesn't exist.
    """
    path = data_path(db_path(name), **kwargs)
    return sqlite3.connect(path)


class TestCLgen(TestCase):
    def test_checksum(self):
        self.assertEqual("0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                         clgen.checksum("foo".encode()))
        self.assertEqual("62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                         clgen.checksum("bar".encode()))

    def test_checksum_str(self):
        self.assertEqual("0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                         clgen.checksum_str("foo"))
        self.assertEqual("62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                         clgen.checksum_str("bar"))
        self.assertEqual("ac3478d69a3c81fa62e60f5c3696165a4e5e6ac4",
                         clgen.checksum_str(5))

    def test_checksum_file(self):
        with self.assertRaises(clgen.InternalError):
            clgen.checksum_file("NOT A PATH")

    def test_get_substring_idxs(self):
        self.assertEqual([0, 2], clgen.get_substring_idxs('a', 'aba'))
        self.assertEqual([], clgen.get_substring_idxs('a', 'bb'))

    def test_pacakge_data(self):
        with self.assertRaises(clgen.InternalError):
            clgen.package_data("This definitely isn't a real path")
        with self.assertRaises(clgen.File404):
            clgen.package_data("This definitely isn't a real path")

    def test_pacakge_str(self):
        with self.assertRaises(clgen.InternalError):
            clgen.package_str("This definitely isn't a real path")
        with self.assertRaises(clgen.File404):
            clgen.package_str("This definitely isn't a real path")

    def test_sql_script(self):
        with self.assertRaises(clgen.InternalError):
            clgen.sql_script("This definitely isn't a real path")
        with self.assertRaises(clgen.File404):
            clgen.sql_script("This definitely isn't a real path")
