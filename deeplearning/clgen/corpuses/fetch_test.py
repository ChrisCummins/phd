"""Unit tests for //deeplearning/clgen/fetch.py."""
import sys

import pytest
from absl import app


def test_TODO():
  pass


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
