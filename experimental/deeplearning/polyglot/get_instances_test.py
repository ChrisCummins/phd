"""Tests for //experimental/deeplearning/polyglot/baselines/get_instances.py."""
import sys
import tempfile

import pytest
from absl import app
from absl import flags

from experimental.deeplearning.polyglot import get_instances


FLAGS = flags.FLAGS


def test_GetInstances():
  """Short summary of test."""
  with tempfile.TemporaryDirectory() as working_dir:
    flags.FLAGS(['argv[0]', '--working_dir', working_dir])
    assert FLAGS.working_dir == working_dir
    assert get_instances.GetInstances()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
