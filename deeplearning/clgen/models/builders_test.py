"""Unit tests for //deeplearning/clgen/models/builders.py."""
import sys

import pytest
from absl import app
from absl import flags


FLAGS = flags.FLAGS


# TODO(cec): Add test where batch_size is larger than corpus.

def test_TODO():
  pass


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
