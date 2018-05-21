"""Pytest fixtures for CLgen unit tests."""
import os
import tempfile

import pytest
from absl import flags


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def clgen_cache_dir() -> str:
  """Creates a temporary directory and sets CLGEN_CACHE to it.

  Returns:
    The location of $CLGEN_CACHE.
  """
  with tempfile.TemporaryDirectory() as d:
    os.environ['CLGEN_CACHE'] = d
    yield d
