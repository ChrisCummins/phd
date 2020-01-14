# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines the base classes and utilities for formatters."""
import os
import pathlib
import subprocess
import sys
from typing import List
from typing import Optional

from labm8.py import app

FLAGS = app.FLAGS


class Formatter(object):
  """Base class for implementing formatters."""

  def __init__(self, cache_path: pathlib.Path):
    self.cache_path = cache_path
    self.futures = []

  def __call__(self, path: pathlib.Path, cached_mtime: Optional[int]):
    app.Log(2, "format %s", path)
    return lambda: ([path], [cached_mtime], self.RunOne(path))

  def RunOne(self, path: pathlib.Path) -> Optional[str]:
    raise NotImplementedError("abstract class")

  def Finalize(self):
    pass


class BatchedFormatter(Formatter):
  """A formatter which processes multiple files in batches."""

  def __init__(self, cache_path: pathlib.Path, batch_size: int = 32):
    super(BatchedFormatter, self).__init__(cache_path)
    self.batch_size = batch_size
    self._actions = []

  def __call__(self, path: pathlib.Path, cached_mtime: Optional[int]):
    app.Log(2, "format %s", path)

    self._actions.append((path, cached_mtime))
    if len(self._actions) > self.batch_size:
      paths = [x[0] for x in self._actions]
      cached_mtimes = [x[1] for x in self._actions]
      action = lambda: (paths, cached_mtimes, self.RunMany(paths))
      self._actions = []
      return action

  def RunMany(self, paths: List[pathlib.Path]) -> Optional[str]:
    raise NotImplementedError

  def Finalize(self):
    if self._actions:
      paths = [x[0] for x in self._actions]
      cached_mtimes = [x[1] for x in self._actions]
      return lambda: (paths, cached_mtimes, self.RunMany(paths))


def WhichOrDie(name: str, install_instructions: Optional[str] = None):
  """Lookup the absolute path of a binary. If not found, abort.

  Args;
    name: The binary to look up.

  Returns:
    The abspath of the binary.
  """
  for path in os.environ["PATH"].split(os.pathsep):
    if os.path.exists(os.path.join(path, name)):
      return os.path.join(path, name)
  print("ERROR: Could not find required binary:", name, file=sys.stderr)
  if install_instructions:
    print(install_instructions, file=sys.stderr)
  else:
    print(
      "You probably haven't installed the development dependencies. "
      "See INSTALL.md.",
      file=sys.stderr,
    )
  sys.exit(1)


def ExecOrError(cmd):
  """Run the given command silently and return its output if it fails."""
  if app.GetVerbosity() >= 3:
    app.Log(3, "exec $ %s", " ".join(str(x) for x in cmd))

  process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
  )
  stdout, _ = process.communicate()
  if process.returncode:
    return stdout
