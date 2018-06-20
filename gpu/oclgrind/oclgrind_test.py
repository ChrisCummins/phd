"""Unit tests for //gpu/oclgrind/oclgrind.py."""
import sys

import pytest
from absl import app
from absl import flags

from gpu.oclgrind import oclgrind


FLAGS = flags.FLAGS

# The verbatim string printed to stdout by `oclgrind --version`.
VERSION = """
Oclgrind 18.3

Copyright (c) 2013-2018
James Price and Simon McIntosh-Smith, University of Bristol
https://github.com/jrprice/Oclgrind

"""


def test_Exec_version():
  """Test that the version of oclgrind is as expected."""
  proc = oclgrind.Exec(['--version'])
  # This test will of course fail if the @oclgrind package is updated.
  assert proc.stdout == VERSION


def test_OpenCLEnvironment_Exec_version():
  """Test that OpenCLEnvironment.Exec() works as expected for version."""
  proc = oclgrind.OpenCLEnvironment().Exec(['--version'])
  # This test will of course fail if the @oclgrind package is updated.
  assert proc.stdout == VERSION


def test_OpenCLEnvironment_name():
  """Test that the OpenCLEnvironment object has a correct 'name' property."""
  env = oclgrind.OpenCLEnvironment()
  # This test will of course fail if the @oclgrind package is updated.
  assert 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2' == env.name


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
