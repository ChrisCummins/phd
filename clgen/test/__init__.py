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
import os
import pytest
import sqlite3
import sys
import tarfile

from io import StringIO
from labm8 import fs
from labm8 import tar
from unittest import TestCase

import clgen


class Data404(Exception):
    pass


def data_path(*components, **kwargs):
    """
    Return absolute path to unittest data file. Data files are located in
    tests/data.

    Args:
        *components (str[]): Relative path.
        **kwargs (dict, optional): If 'exists' True, require that file exists.

    Returns:
        string: Absolute path.

    Raises:
        Data404: If path doesn"t exist.
    """
    path = fs.path(*components)
    exists = kwargs.get("exists", True)

    abspath = os.path.join(os.path.dirname(__file__), "data", path)
    if exists and not os.path.exists(abspath):
        raise Data404(abspath)
    return abspath


def data_str(*components):
    """
    Return contents of unittest data file as a string.

    Args:
        *components (str[]): Relative path.

    Returns:
        string: File contents.

    Raises:
        Data404: If path doesn't exist.
    """
    path = fs.path(*components)

    with open(data_path(path)) as infile:
        return infile.read()


def archive(*components):
    """
    Returns a text archive, unpacking if necessary.

    Arguments:
        *components (str[]): Relative path.

    Returns:
        str: Path to archive.
    """
    path = data_path(*components, exists=False)

    if not fs.isdir(path):
        tar.unpack_archive(path + ".tar.bz2")
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
        Data404: If path doesn't exist.
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
        Data404: If path doesn't exist.
    """
    path = data_path(db_path(name), **kwargs)
    return sqlite3.connect(path)


def local_cachepath(*relative_path_components: list) -> str:
    """ return path to local testing cache """
    assert(relative_path_components)

    cache_root = [data_path("cache", exists=False)]
    return fs.path(*cache_root, *relative_path_components)

# use local cache for testing
clgen.cachepath = local_cachepath


class TestCLgen(TestCase):
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

    def test_platform_info(self):
        clgen.platform_info()


class DevNullRedirect(object):
    """
    Context manager to redirect stdout and stderr to devnull.

    Examples
    --------
    >>> with DevNullRedirect(): print("this will not print")
    """
    def __init__(self):
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def main():
    # run from module directory
    module_path = os.path.dirname(clgen.__file__)
    os.chdir(module_path)

    pytest.main(["--doctest-modules", "--cov=clgen",
                 "--cov-config", ".coveragerc"])
