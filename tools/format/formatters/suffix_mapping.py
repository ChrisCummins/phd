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
"""This module defines the mapping from file names to formatters."""
from labm8.py import app
from tools.format.formatters import bazel
from tools.format.formatters import cxx
from tools.format.formatters import go
from tools.format.formatters import java
from tools.format.formatters import javascript
from tools.format.formatters import json
from tools.format.formatters import python
from tools.format.formatters import sql
from tools.format.formatters import text

FLAGS = app.FLAGS

# The mapping from path suffixes to Formatter classes. To look up the formatter
# for a path, key into this dictionary first by pathlib.Path.suffix, or by
# pathlib.Path.name if there is no suffix.
mapping = {
  ".bzl": python.FormatPython,
  ".c": cxx.FormatCxx,
  ".cc": cxx.FormatCxx,
  ".cpp": cxx.FormatCxx,
  ".css": javascript.FormatJavaScript,
  ".cxx": cxx.FormatCxx,
  ".formatignore": text.FormatText,
  ".go": go.FormatGo,
  ".h": cxx.FormatCxx,
  ".hpp": cxx.FormatCxx,
  ".html": javascript.FormatJavaScript,
  ".ino": cxx.FormatCxx,
  ".java": java.FormatJava,
  ".js": javascript.FormatJavaScript,
  ".json": json.FormatJson,
  ".md": text.FormatText,
  ".py": python.FormatPython,
  ".sql": sql.FormatSql,
  ".txt": text.FormatText,
  "BUILD": bazel.FormatBuild,
  "WORKSPACE": bazel.FormatBuild,
}
