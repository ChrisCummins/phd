"""Unit tests for //lib/labm8/dirhashcache.py."""
import sys

import pytest
from absl import app
from absl import flags

from lib.labm8 import dirhashcache


FLAGS = flags.FLAGS


def test_DirHashCache_invalid_hash_function():
  with pytest.raises(ValueError):
    dirhashcache.DirHashCache('.', hash_function='not a valid hash function')


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
