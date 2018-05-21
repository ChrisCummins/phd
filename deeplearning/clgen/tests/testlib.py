"""Shared code for testing CLgen."""
import contextlib
import os
import sqlite3
import sys
from io import StringIO
from pathlib import Path

import pytest
from absl import flags

from deeplearning.clgen import clgen_config as config
from lib.labm8 import fs
from lib.labm8 import system
from lib.labm8 import tar


FLAGS = flags.FLAGS


class Data404(Exception):
  pass


needs_cuda = pytest.mark.skipif(not config.USE_CUDA, reason="no CUDA support")
needs_linux = pytest.mark.skipif(not system.is_linux(), reason="not linux")
skip_on_travis = pytest.mark.skipif(os.environ.get("TRAVIS") == 'true',
                                    reason="skip on Travis CI")


def data_path(*components, exists=True) -> str:
  """
  Return absolute path to unittest data file. Data files are located in
  <package>/test/data.

  Parameters
  ----------
  *components : str
      Relative path.
  exists : bool, optional
      If True, require that file exists.

  Returns
  -------
  str
      Absolute path.

  Raises
  ------
  Data404
      If path doesn't exist and 'exists' is True.
  """
  path = fs.path(*components)

  abspath = os.path.join(os.path.dirname(__file__), "data", path)
  if exists and not os.path.exists(abspath):
    raise Data404(abspath)
  return abspath


def data_str(*components) -> str:
  """
  Return contents of unittest data file as a string.

  Parameters
  ----------
  *components : str
      Relative path.

  Returns
  -------
  str
      File contents.

  Raises
  ------
  Data404
      If path doesn't exist.
  """
  path = data_path(*components, exists=True)

  with open(data_path(path)) as infile:
    return infile.read()


def archive(*components):
  """
  Returns a text archive, unpacking if necessary.

  Parameters
  ----------
  *components : str
      Relative path.

  Returns
  -------
  str
      Path to archive.
  """
  path = data_path(*components, exists=False)

  if not fs.isdir(path):
    tar.unpack_archive(path + ".tar.bz2")
  return path


def db_path(path):
  """
  Return absolute path to unittest data file. Data files are located in
  <package>/test/data/db.

  Parameters
  ----------
  path : str
      Relative path.

  Returns
  -------
  str
      Absolute path.

  Raises
  ------
  Data404
      If path doesn't exist.
  """
  return data_path(os.path.join("db", str(path) + ".db"))


def db(name, **kwargs):
  """
  Return absolute path to unittest data file. Data files are located in
  tests/data/db.

  Parameters
  ----------
  path : str
      Relative path.

  Returns
  -------
  sqlite.Connection
      Sqlite connection to database.

  Raises
  ------
  Data404
      If path doesn't exist.
  """
  path = data_path(db_path(name), **kwargs)
  return sqlite3.connect(path)


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


@contextlib.contextmanager
def chdir(path: Path):
  """
  Changes working directory and returns to previous on exit

  By @Lukas http://stackoverflow.com/a/42441759
  """
  prev_cwd = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(prev_cwd)


def module_path():
  return os.path.dirname(clgen.__file__)


def test_cache_path():
  return data_path("cache", exists=False)


def coverage_report_path():
  return os.path.join(module_path(), ".coverage")


def coveragerc_path():
  return data_path("coveragerc")


@contextlib.contextmanager
def test_env():
  """
  Manages the environment used for tests.
  """
  # use local cache for testing
  old_cachepath = os.environ.get("CLGEN_CACHE")
  os.environ["CLGEN_CACHE"] = test_cache_path()

  # no GPUs for testing
  old_cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES")
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  try:
    yield
  finally:
    if old_cachepath:
      os.environ["CLGEN_CACHE"] = old_cachepath
    if old_cuda_devs:
      os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_devs


def testsuite():
  """
  Run the CLgen test suite.

  Returns
  -------
  int
      Test return code. 0 if successful.
  """
  with test_env():
    with chdir(module_path()):  # run from module directory
      assert os.path.exists(coveragerc_path())

      args = ["--doctest-modules", "--cov=clgen", "--cov-config",
              coveragerc_path()]

      # unless verbose, don't print coverage report
      if log.is_verbose():
        args.append("--verbose")
      else:
        args.append("--cov-report=")

      ret = pytest.main(args)

      assert os.path.exists(coverage_report_path())

    if log.is_verbose():
      print("coverage path:", coverage_report_path())
      print("coveragerc path:", coveragerc_path())

  return ret
