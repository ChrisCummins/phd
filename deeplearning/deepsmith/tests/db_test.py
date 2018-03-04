"""Unit tests for db.py."""
import sys

import pytest
from absl import app

from deeplearning.deepsmith import db

def test_HelloWorld():
  assert 2 == 1 + 1


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
