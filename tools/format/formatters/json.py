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
import json

from tools.format.formatters.base import file_formatter


class FormatJson(file_formatter.FileFormatter):
  """Format JSON files."""

  def RunOne(self, path):
    """Format a json file."""
    # First, read the entire file into memory, then parse the JSON and serialize
    # it a formatted string. Compare the input and output strings and, if they
    # are not equal, write the formatted string.
    #
    # We do this comparison rather than unconditionally writing the formatted
    # JSON output because otherwise the mtime of this file would always change
    # with every call to this function, even if the file contents remain the
    # same. This would create unnecessary mtime cache misses.
    with open(path, "r") as f:
      text = f.read()
    try:
      data = json.loads(text)
    except json.decoder.JSONDecodeError as e:
      raise self.FormatError(
        f"Failed to parse JSON: {path}\n    Parser error: {e}"
      )
    formatted_text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if text != formatted_text:
      with open(path, "w") as f:
        f.write(formatted_text)
