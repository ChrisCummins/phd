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
"""This module defines a formatter for shell (i.e. bash, bats) sources."""
import sys

from labm8.py import bazelutil
from tools.format.formatters.base import batched_file_formatter


class FormatShell(batched_file_formatter.BatchedFileFormatter):
  """Format shell sources."""

  def __init__(self, *args, **kwargs):
    super(FormatShell, self).__init__(*args, **kwargs)
    arch = "mac" if sys.platform == "darwin" else "linux"
    self.shfmt = bazelutil.DataPath(f"shfmt_{arch}/file/shfmt")

  def RunMany(self, paths):
    # To enable shfmt to parse bats tests we must insert a newline before the
    # opening brace.
    # See https://www.gitmemory.com/issue/bats-core/bats-core/192/528315083
    bats_paths = [p for p in paths if p.suffix == ".bats"]
    if bats_paths:
      error = self._Exec(
        ["perl", "-pi", "-e", "s/^(\@test.*) \{$/$1\n{/"] + bats_paths
      )
      if error:
        return error

    error = self._Exec([self.shfmt, "-i", "2", "-ci", "-w"] + paths)
    if error:
      return error

    # Reverse the temporary reformatting of bats tests.
    if bats_paths:
      error = self._Exec(
        ["perl", "-pi", "-e", "s/^\{\R//; s/(\@test.*$)/$1 {/"] + bats_paths
      )
      if error:
        return error
