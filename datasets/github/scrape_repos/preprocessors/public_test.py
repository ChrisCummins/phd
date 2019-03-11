"""Unit tests for //datasets/github/scrape_repos/public.py."""
import pathlib

from datasets.github.scrape_repos.preprocessors import public
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

# GetAllFilesRelativePaths() tests.


def test_GetAllFilesRelativePaths_empty_dir(tempdir: pathlib.Path):
  """Test that an empty directory returns an empty list."""
  assert public.GetAllFilesRelativePaths(tempdir) == []


def test_GetAllFilesRelativePaths_relpath(tempdir: pathlib.Path):
  """Test that relative paths are returned."""
  (tempdir / 'a').touch()
  assert public.GetAllFilesRelativePaths(tempdir) == ['a']


if __name__ == '__main__':
  test.Main()
