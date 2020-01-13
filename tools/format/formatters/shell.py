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
import os
import sys

from labm8.py import bazelutil
from tools.format import formatter


class FormatShell(formatter.BatchedFormatter):
  """Format shell sources."""

  def __init__(self, *args, **kwargs):
    super(FormatShell, self).__init__(*args, **kwargs)

    # Unpack shfmt.
    self.shfmt = self.cache_path / "shfmt"
    if not self.shfmt.is_file():
      if sys.platform == "darwin":
        shfmt = bazelutil.DataString("shfmt_mac/file/shfmt")
      else:
        shfmt = bazelutil.DataString("shfmt_linux/file/shfmt")
      with open(self.shfmt, "wb") as f:
        f.write(shfmt)
      os.chmod(self.shfmt, 0o744)

  def RunMany(self, paths):
    return formatter.ExecOrError([self.shfmt, "-i", "2", "-ci", "-w"] + paths)
