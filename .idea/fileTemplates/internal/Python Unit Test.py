"""Tests for //TODO:${NAME}."""
import sys

import pytest
from absl import app
from absl import flags


FLAGS = flags.FLAGS


def test_TODO() -> None:
  # TODO: Short summary of the test.
  pass


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unknown flags "{}".'.format(', '.join(argv[1:])))
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == '__main__':
  app.run(main)
