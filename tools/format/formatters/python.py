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
"""This module defines a formatter for Python sources."""
from tools.format.formatters import formatter


class FormatPython(formatter.BatchedFormatter):
  """Format Python sources."""

  def __init__(self, *args, **kwargs):
    super(FormatPython, self).__init__(*args, **kwargs)
    self.black = formatter.WhichOrDie("black")
    self.reorder_python_imports = formatter.WhichOrDie("reorder-python-imports")

  def RunMany(self, paths):
    error = formatter.ExecOrError(
      [self.black, "--line-length=80", "--target-version=py37"] + paths
    )
    if error:
      return error
    return formatter.ExecOrError(
      [self.reorder_python_imports, "--exit-zero-even-if-changed"] + paths
    )
