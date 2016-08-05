from unittest import TestCase

import sqlite3
import os

import smith

class TestDataNotFoundException(Exception): pass


def data_path(path, exists=True):
    """
    Return path to unittest data file. Data files are located in
    tests/data.

    :param path: Relative path (without 'data/path')
    :return: String path.
    :throws TestDataNotFoundException: If file doesn't exist.
    """
    abspath = os.path.join(os.path.dirname(__file__), 'data', path)
    if exists and not os.path.exists(abspath):
        raise TestDataNotFoundException(abspath)
    return abspath


def data_str(path):
    """
    Return contents of unittest data file as a string.

    :param path: Path to data file.
    :return: File contents as string.
    """
    with open(data_path(path)) as infile:
        contents = infile.read()
    return contents


def db(name, **kwargs):
    """
    """
    path = data_path(os.path.join('db', str(name) + '.db'), **kwargs)
    return sqlite3.connect(path)


class TestSmith(TestCase):
    def test_checksum(self):
        self.assertEqual("0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                          smith.checksum("foo".encode()))
        self.assertEqual("62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                          smith.checksum("bar".encode()))

    def test_checksum_str(self):
        self.assertEqual("0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                          smith.checksum_str("foo"))
        self.assertEqual("62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                          smith.checksum_str("bar"))
        self.assertEqual("ac3478d69a3c81fa62e60f5c3696165a4e5e6ac4",
                          smith.checksum_str(5))

    def test_checksum_file(self):
        with self.assertRaises(smith.InternalException):
            smith.checksum_file("NOT A PATH")

    def test_get_substring_idxs(self):
        self.assertEqual([0, 2], smith.get_substring_idxs('a', 'aba'))
        self.assertEqual([], smith.get_substring_idxs('a', 'bb'))

    def test_pacakge_data(self):
        with self.assertRaises(smith.InternalException):
            smith.package_data("This definitely isn't a real path")
        with self.assertRaises(smith.Data404Exception):
            smith.package_data("This definitely isn't a real path")

    def test_pacakge_str(self):
        with self.assertRaises(smith.InternalException):
            smith.package_str("This definitely isn't a real path")
        with self.assertRaises(smith.Data404Exception):
            smith.package_str("This definitely isn't a real path")

    def test_sql_script(self):
        with self.assertRaises(smith.InternalException):
            smith.sql_script("This definitely isn't a real path")
        with self.assertRaises(smith.Data404Exception):
            smith.sql_script("This definitely isn't a real path")


if __name__ == '__main__':
    main()
