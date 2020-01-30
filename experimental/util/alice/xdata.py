# Copyright (c) 2019-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# alice is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alice.  If not, see <https://www.gnu.org/licenses/>.
"""TODO: Experimental data API."""
import contextlib
import os
import pathlib

from labm8.py import app

FLAGS = app.FLAGS


class NotInWonderland(EnvironmentError):
  pass


def Id() -> int:
  id = os.environ.get("ALICE_XDATA_ID")
  if not id:
    raise NotImplementedError
  return id


def CreateArtifactDirectory(name: str) -> pathlib.Path:
  pass


class JsonWriter(object):
  pass


@contextlib.contextmanager
def TemporaryInMemoryDirectory(name: str) -> pathlib.Path:
  pass


@contextlib.contextmanager
def TemporaryDirectory(name: str) -> pathlib.Path:
  pass
