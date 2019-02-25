"""A test python file."""
import sys

import pytest
from absl import app
from absl import flags
from absl import logging

from labm8 import bazelutil

FLAGS = flags.FLAGS


def test_datafile_read():
  """Test reading a data file."""
  with open(bazelutil.DataPath('phd/learn/docker/bazel/datafile.txt')) as f:
    assert f.read() == 'Hello, Docker!\n'


def main(argv):
  """Main entry point."""
  logging.info('Platform: %s', sys.platform)
  logging.info('Exec:     %s', sys.executable)
  logging.info('Args:     %s', ' '.join(argv))
  if len(argv) > 1:
    logging.warning("Unknown arguments: '%s'", ' '.join(argv[1:]))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
