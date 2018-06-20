"""Unit tests for //gpu/oclgrind/oclgrind.py."""
import sys

import pytest
from absl import app
from absl import flags

from gpu.oclgrind import oclgrind


FLAGS = flags.FLAGS


def test_Exec_version():
  """Test that the version of oclgrind is as expected."""
  proc = oclgrind.Exec(['--version'])
  # This test will of course fail if the @oclgrind package is updated.
  assert proc.stdout == """
Oclgrind 18.3

Copyright (c) 2013-2018
James Price and Simon McIntosh-Smith, University of Bristol
https://github.com/jrprice/Oclgrind

"""


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
