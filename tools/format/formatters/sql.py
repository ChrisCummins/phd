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
"""This module defines a formatter for SQL sources."""
import sqlparse

from labm8.py import fs
from tools.format import formatter


class FormatSql(formatter.Formatter):
  """Format SQL files."""

  def __init__(self, *args, **kwargs):
    super(FormatSql, self).__init__(*args, **kwargs)

  def RunOne(self, path):
    try:
      formatted = (
        sqlparse.format(
          fs.Read(path), reindent=True, keyword_case="upper"
        ).rstrip()
        + "\n"
      )
    except:
      return f"sqlparse failed for: {path}"

    fs.Write(path, formatted.encode("utf-8"))
