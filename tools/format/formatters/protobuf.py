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
"""This module defines a formatter for protocol buffers."""
import os
import pathlib
import sys

from labm8.py import bazelutil
from tools.format.formatters.base import file_formatter


class FormatProtobuf(file_formatter.FileFormatter):
  """Format protocol buffer sources.

  This uses prototool's format command to automatically format proto files.
  Although the prototool recommends switching to `buf`, buf does not yet have
  a formatter, see: https://buf.build/docs/lint-checkers#formatting.

  Currently I also run a pass of prototool's lint command to enforce linting
  rules. In the future I would like to switch to buf's linter as this seems to
  have a nicer and broader rule set, but I'm deferring this task for now.
  """

  assumed_filename = "input.proto"

  def __init__(self, *args, **kwargs):
    super(FormatProtobuf, self).__init__(*args, **kwargs)

    self.prototool = bazelutil.DataPath(
      f"prototool_{sys.platform}/file/prototool"
    )

    # Make a local cache for prototool, since otherwise it will try to write to
    # $HOME.
    self.prototool_cache = self.cache_path / "prototool"
    self.prototool_cache.mkdir(exist_ok=True)

  @staticmethod
  def GuessRootDirectory(path: pathlib.Path) -> pathlib.Path:
    """Guess the root directory for a path to a proto.

    This looks for a bazel WORKSPACE in the parent directories, and if found,
    uses that.
    """
    for parent in path.parents:
      if (parent / "WORKSPACE").is_file():
        return parent

    # Fallback: use the directory containing the proto.
    return path.parent

  def RunOne(self, path: pathlib.Path) -> None:
    # Prototool is sensitive to the directory in which it is run. This is
    # because it needs to compile the proto file, so must be able to resolve the
    # paths of "import" statements.
    previous_wd = os.getcwd()
    os.chdir(self.GuessRootDirectory(path))
    try:
      self._Exec(
        [
          self.prototool,
          "--cache-path",
          self.prototool_cache,
          '--config-data={"lint": {"group": "google"}}',
          "lint",
          path,
        ]
      )

      self._Exec(
        [
          self.prototool,
          "--cache-path",
          self.prototool_cache,
          "format",
          "-w",
          "-f",
          path,
        ]
      )
    finally:
      os.chdir(previous_wd)
