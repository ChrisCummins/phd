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
"""Utilities for testing formatters."""
import pathlib
import tempfile

from labm8.py import fs


def FormatText(formatter_class, text: str, filename: str = "text") -> str:
  with tempfile.TemporaryDirectory(prefix="format_test_") as d:
    cache_path = pathlib.Path(d)
    formatter = formatter_class(cache_path)

    path = cache_path / filename
    fs.Write(path, text.encode("utf-8"))

    actions = [formatter(path, cached_mtime=None), formatter.Finalize()]

    # Run the actions and accumulate errors.
    errors = []
    for action in actions:
      if action:
        _, _, error = action()
        if error:
          errors.append(error)

    if errors:
      raise ValueError("\n".join(errors))

    return fs.Read(path)
