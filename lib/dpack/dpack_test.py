"""Unit tests for //lib/dpack:dpack."""
import sys

import pytest
from absl import app

from lib.dpack import dpack


def test_SetDataPackageFileAttributes():
  pass


def test_CreatePackageManifest():
  dpack.CreatePackageManifest()


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
