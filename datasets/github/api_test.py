"""Unit tests for //datasets/github:api.py"""
import pathlib
import tempfile

import pytest
from absl import flags

from datasets.github import api
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def credentials_file() -> pathlib.Path:
  """A test fixture to yield a GitHub credentials file."""
  with tempfile.TemporaryDirectory() as d:
    with open(pathlib.Path(d) / 'credentials', 'w') as f:
      f.write("""
[User]
Username = foo
Password = bar
""")
    yield pathlib.Path(d) / 'credentials'


def test_ReadGitHubCredentials(credentials_file: pathlib.Path):
  """Test that GitHub credentials are read from the filesystem."""
  credentials = api.ReadGitHubCredentials(credentials_file)
  assert credentials.HasField('username')
  assert credentials.username == 'foo'
  assert credentials.HasField('password')
  assert credentials.password == 'bar'


if __name__ == '__main__':
  test.Main()
