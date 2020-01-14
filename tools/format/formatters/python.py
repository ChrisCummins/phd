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
from tools.format import formatter
import black
from click.testing import CliRunner


class FormatPython(formatter.BatchedFormatter):
  """Format Python sources."""

  def __init__(self, *args, **kwargs):
    super(FormatPython, self).__init__(*args, **kwargs)
    self.reorder_python_imports = formatter.WhichOrDie("reorder-python-imports")

    # Replicate functionality of black.patched_main().
    black.patch_click()

    # Create a runner for the black command line.
    self.black_runner = CliRunner()

  def RunMany(self, paths):
    # Invoke black.
    result = self.black_runner.invoke(
        black.main, ["--line-length=80", "--target-version=py37"] + [
            str(x) for x in paths
        ])
    if result.exit_code:
      return result.output

    return formatter.ExecOrError(
      [self.reorder_python_imports, "--exit-zero-even-if-changed"] + paths
    )
