"""Unit tests for //datasets/github/scrape_repos/public.py."""
import pathlib
import sys

import pytest
from absl import app
from absl import flags

from datasets.github.scrape_repos.preprocessors import public


FLAGS = flags.FLAGS


# GetAllFilesRelativePaths() tests.

def test_GetAllFilesRelativePaths_empty_dir(tempdir: pathlib.Path):
  """Test that an empty directory returns an empty list."""
  assert public.GetAllFilesRelativePaths(tempdir) == []


def test_GetAllFilesRelativePaths_relpath(tempdir: pathlib.Path):
  """Test that relative paths are returned."""
  (tempdir / 'a').touch()
  assert public.GetAllFilesRelativePaths(tempdir) == ['a']


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
