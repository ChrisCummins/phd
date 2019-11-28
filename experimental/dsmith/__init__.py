#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Compiler fuzzing through deep learning.

Attributes:
    __version__ (str): PEP 440 compliant version string.
    version_info (Tuple[int, int, int, str]): Major, minor, micro, and
        releaselevel version components.
    DB_HOSTNAME (str): Hostname of database server.
    DB_PORT (int): Database server port.
    DB_CREDENTIALS (Tuple[str, str]): Database username and password.
    DB_BUF_SIZE (int): Number of records to buffer before adding to database.
"""
import logging
import os
import re
from collections import namedtuple
from configparser import ConfigParser
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

from pkg_resources import require
from pkg_resources import resource_filename

from experimental.dsmith._config import *
from labm8.py import fs

__author__ = "Chris Cummins"
__copyright__ = "Copyright 2017, 2018 Chris Cummins"
__license__ = "GPL v3"
__version__ = require("dsmith")[0].version
__maintainer__ = __author__
__email__ = "chrisc.101@gmail.com"
__status__ = "Prototype"

__version_str__ = (
  f"dsmith {__version__} made with \033[1;31m♥\033[0;0m by "
  "Chris Cummins <chrisc.101@gmail.com>."
)

# version_info tuple
_major = int(__version__.split(".")[0])
_minor = (
  int(__version__.split(".")[1]) if len(__version__.split(".")) > 1 else 0
)
_micro = (
  int(__version__.split(".")[2]) if len(__version__.split(".")) > 2 else 0
)
_releaselevel = (
  __version__.split(".")[3] if len(__version__.split(".")) > 3 else "final"
)

version_info_t = namedtuple(
  "version_info_t", ["major", "minor", "micro", "releaselevel"]
)
version_info = version_info_t(_major, _minor, _micro, _releaselevel)

# set by init_globals()
DB_ENGINE = None
DB_HOSTNAME = None
DB_PORT = None
DB_CREDENTIALS = None
DB_DIR = None
DB_BUF_SIZE = None


def init_globals(rc_path: Path) -> None:
  global DB_ENGINE
  global DB_HOSTNAME
  global DB_PORT
  global DB_CREDENTIALS
  global DB_DIR
  global DB_BUF_SIZE

  path = fs.abspath(rc_path)

  _config = ConfigParser()
  _config.read(path)
  DB_ENGINE = _config["database"]["engine"].lower()
  DB_HOSTNAME = _config["database"].get("hostname", "")
  DB_PORT = _config["database"].get("port", "")
  DB_CREDENTIALS = (
    _config["database"].get("username", ""),
    _config["database"].get("password", ""),
  )
  DB_DIR = _config["database"].get("dir", "")
  DB_BUF_SIZE = int(_config["database"]["buffer_size"])


init_globals(RC_PATH)


class DSmithError(Exception):
  """
  Top level error. Never directly thrown.
  """

  pass


class InternalError(DSmithError):
  """
  An internal module error. This class of errors should not leak outside of
  the module into user code.
  """

  pass


class UserError(DSmithError):
  """
  Raised in case of bad user interaction, e.g. an invalid argument.
  """

  pass


class Filesystem404(InternalError):
  """
  Path not found.
  """

  pass


class Colors:
  PURPLE = "\033[95m"
  CYAN = "\033[96m"
  DARKCYAN = "\033[36m"
  BLUE = "\033[94m"
  GREEN = "\033[92m"
  YELLOW = "\033[93m"
  RED = "\033[91m"
  BOLD = "\033[1m"
  UNDERLINE = "\033[4m"
  END = "\033[0m"


def data_path(*path) -> Path:
  """
  Path to data file.

  Arguments:
      *path (List[str]): Path components.

  Returns:
      Path: Path.
  """
  return resource_filename(__name__, fs.path("data", *path))


def root_path(*path) -> Path:
  """
  Path relative to dsmith source respository.

  Arguments:
      *path (List[str]): Path components.

  Returns:
      Path: Path.
  """
  return fs.path(ROOT, *path)


@contextmanager
def debug_scope() -> None:
  """ Provide a scope for running operations with debugging output. """
  old_debug_level = logging.getLogger("").level
  old_debug_env = os.environ.get("DEBUG", "")

  logging.getLogger("").setLevel(logging.DEBUG)
  os.environ["DEBUG"] = "1"

  try:
    yield
  finally:
    logging.getLogger("").setLevel(old_debug_level)
    os.environ["DEBUG"] = old_debug_env


@contextmanager
def verbose_scope() -> None:
  """ Provide a scope for running operations with verbose output. """
  old_debug_level = logging.getLogger("").level

  logging.getLogger("").setLevel(logging.INFO)

  try:
    yield
  finally:
    logging.getLogger("").setLevel(old_debug_level)


def unformat(string: str) -> str:
  """ strip shell formatting escape codes """
  return re.sub(r"\x1b[^m]*m", "", string.split(" ")[0])
