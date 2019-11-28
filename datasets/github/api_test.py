# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //datasets/github:api.py"""
import pathlib
import tempfile

import pytest

from datasets.github import api
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope="function")
def credentials_file() -> pathlib.Path:
  """A test fixture to yield a GitHub credentials file."""
  with tempfile.TemporaryDirectory() as d:
    with open(pathlib.Path(d) / "credentials", "w") as f:
      f.write(
        """
[User]
Username = foo
Password = bar
"""
      )
    yield pathlib.Path(d) / "credentials"


def test_ReadGitHubCredentials(credentials_file: pathlib.Path):
  """Test that GitHub credentials are read from the filesystem."""
  credentials = api.ReadGitHubCredentials(credentials_file)
  assert credentials.HasField("username")
  assert credentials.username == "foo"
  assert credentials.HasField("password")
  assert credentials.password == "bar"


if __name__ == "__main__":
  test.Main()
