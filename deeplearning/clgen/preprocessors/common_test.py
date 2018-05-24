"""Unit tests for ///common_test.py."""
import sys

import pytest
from absl import app
from absl import flags


FLAGS = flags.FLAGS


def test_TODO():
  pass


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
