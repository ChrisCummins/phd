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
"""This module defines a formatter for go sources."""
import os
import sys

from labm8.py import bazelutil
from tools.format import formatter


class FormatGo(formatter.Formatter):
  """Format go sources.

  Run linter on each file individually because:
    1. An error in one file prevents linting in all other files.
    2. All files in a single invocation must be in the same directory.
  """

  def __init__(self, *args, **kwargs):
    super(FormatGo, self).__init__(*args, **kwargs)
    self.gofmt = formatter.WhichOrDie("gofmt")

    # Unpack gofmt.
    self.gofmt = self.cache_path / "gofmt"
    if not self.gofmt.is_file():
      arch = "mac" if sys.platform == "darwin" else "linux"
      gofmt = bazelutil.DataString(f"go_{arch}/bin/gofmt")
      with open(self.gofmt, "wb") as f:
        f.write(gofmt)
        os.chmod(self.gofmt, 0o744)

  def RunOne(self, path):
    return formatter.ExecOrError([self.gofmt, "-w", path])
