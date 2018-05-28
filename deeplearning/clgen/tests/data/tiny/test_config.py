"""Test that //deeplearning/clgen/tests/data/tiny/config.pbtxt is valid."""
import pytest
import sys
from absl import app

from deeplearning.clgen import clgen
from lib.labm8 import bazelutil
from lib.labm8 import pbutil


def test_config_is_valid():
  """Test that config proto is valid."""
  clgen.Instance(pbutil.FromFile(
      bazelutil.DataPath('phd/deeplearning/clgen/tests/tiny/config.pbtxt')))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
