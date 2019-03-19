"""Smoke test for //third_party/py/git."""
import pathlib

import pytest

from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_Git(tempdir: pathlib.Path):
  """Test that git module can be imported."""
  import git
  with pytest.raises(git.InvalidGitRepositoryError):
    git.Repo(str(tempdir))


if __name__ == '__main__':
  test.Main()
