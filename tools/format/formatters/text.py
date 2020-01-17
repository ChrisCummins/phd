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
"""This module defines a formatter for JSON files."""
import subprocess
import tempfile

from tools.format.formatters.base import file_formatter


class FormatText(file_formatter.FileFormatter):
  """Format text files."""

  def __init__(self, *args, **kwargs):
    super(FormatText, self).__init__(*args, **kwargs)
    self.sed = self._Which("sed")

  def RunOne(self, path):
    with tempfile.NamedTemporaryFile(dir=self.cache_path) as tmpfile:
      add_newline = subprocess.Popen(
        [self.sed, "-e", "$a\\"],
        stdout=tmpfile,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.PIPE,
      )
      strip_trailing_whitespace = subprocess.Popen(
        [self.sed, "s/[[:space:]]*$//", str(path)],
        stdout=add_newline.stdin,
        stderr=subprocess.DEVNULL,
      )
      add_newline.communicate()

      tmpfile.seek(0)
      with open(path, "wb") as f:
        f.write(tmpfile.read())

      if strip_trailing_whitespace.returncode:
        return f"sed 's/[[:space:]]*$//' failed for: {path}"
      if add_newline.returncode:
        return f"sed -e '$a\\' failed for: {path}"
