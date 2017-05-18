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
from labm8 import system
from labm8 import tar

import clgen
from clgen import log


class Data404(Exception):
    pass


# test decorators
needs_cuda = pytest.mark.skipif(not clgen.USE_CUDA, reason="no CUDA support")
needs_linux = pytest.mark.skipif(not system.is_linux(), reason="not linux")
skip_on_travis = pytest.mark.skipif(
    os.environ.get("TRAVIS") == 'true', reason="skip on Travis CI")


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


def test_cachepath(*relative_path_components: list) -> str:
    """ return path to local testing cache """
    cache_root = data_path("cache", exists=False)
    fs.mkdir(cache_root)

    return fs.path(cache_root, *relative_path_components)


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


def module_path():
    return os.path.dirname(clgen.__file__)


def coverage_report_path():
    return os.path.join(module_path(), ".coverage")


def coveragerc_path():
    return data_path("coveragerc")


def testsuite():
    """
    Run the CLgen test suite.

    Returns
    -------
    int
        Test return code. 0 if successful.
    """
    # use local cache for testing
    old_cachepath = clgen.cachepath
    clgen.cachepath = test_cachepath

    # no GPUs for testing
    old_cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # run from module directory
    cwd = os.getcwd()
    os.chdir(module_path())

    assert os.path.exists(coveragerc_path())

    args = ["--doctest-modules", "--cov=clgen",
            "--cov-config", coveragerc_path()]

    # unless verbose, don't print coverage report
    if log.is_verbose():
        args.append("--verbose")
    else:
        args.append("--cov-report=")

    ret = pytest.main(args)

    assert os.path.exists(coverage_report_path())

    # change back to previous directory
    os.chdir(cwd)

    if log.is_verbose():
        print("coverage path:", coverage_report_path())
        print("coveragerc path:", coveragerc_path())

    # restore cachepath
    clgen.cachepath = old_cachepath
    # restore GPUs
    os.environ = old_cuda_devs

    return ret
