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
"""This module defines a formatter for bazel files."""
import sys

from labm8.py import bazelutil
from tools.format.formatters.base import batched_file_formatter


class FormatBuild(batched_file_formatter.BatchedFileFormatter):
  """Format Bazel BUILD and WORKSPACE files."""

  def __init__(self, *args, **kwargs):
    super(FormatBuild, self).__init__(*args, **kwargs)

    # Unpack buildifier.
    arch = "darwin" if sys.platform == "darwin" else "linux"
    self.buildifier = bazelutil.DataPath(
      f"com_github_bazelbuild_buildtools/buildifier/{arch}_amd64_stripped/buildifier"
    )

  def RunMany(self, paths):
    self._Exec([self.buildifier] + paths)
