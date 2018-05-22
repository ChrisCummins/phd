"""Unit tests for //deeplearning/clgen/fetch.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import fetch
from deeplearning.clgen.tests import testlib as tests


def test_inline_fs_headers():
  src = fetch.inline_fs_headers(tests.data_path("cl", "sample-3.cl"), [])
  assert "MY_DATA_TYPE" in src
  assert "__kernel void" in src


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
