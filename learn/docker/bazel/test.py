"""A test python file."""
import sys

import pytest
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS


def test_hello_world():
  """Test always passes."""
  assert True


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    logging.warning("Unknown arguments: '%s'", ' '.join(argv[1:]))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
