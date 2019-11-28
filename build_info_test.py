# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //:build_info."""
import re

import build_info
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_FormatShortRevision():
  revision = build_info.FormatShortRevision()
  assert re.match(r"[0-9a-f]{7}\*?", revision)


def test_FormatShortRevision_html():
  revision = build_info.FormatShortRevision(html=True)
  assert re.match(r'<a href=".+">[0-9a-f]{7}\*?</a>', revision)


def test_FormatShortBuildDescription():
  description = build_info.FormatShortBuildDescription()
  assert re.match(
    r"build: [0-9a-f]{7}\*? on [0-9]{4}-[0-9]{2}-[0-9]{2} by .+@.+", description
  )


if __name__ == "__main__":
  test.Main()
