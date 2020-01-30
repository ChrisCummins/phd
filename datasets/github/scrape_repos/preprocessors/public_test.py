# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //datasets/github/scrape_repos/public.py."""
import pathlib

from datasets.github.scrape_repos.preprocessors import public
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

# GetAllFilesRelativePaths() tests.


def test_GetAllFilesRelativePaths_empty_dir(tempdir: pathlib.Path):
  """Test that an empty directory returns an empty list."""
  assert public.GetAllFilesRelativePaths(tempdir) == []


def test_GetAllFilesRelativePaths_relpath(tempdir: pathlib.Path):
  """Test that relative paths are returned."""
  (tempdir / "a").touch()
  assert public.GetAllFilesRelativePaths(tempdir) == ["a"]


if __name__ == "__main__":
  test.Main()
