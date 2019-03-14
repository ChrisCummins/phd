"""Unit tests for //gpu/oclgrind/oclgrind.py."""

from gpu.oclgrind import oclgrind
from labm8 import app
from labm8 import bazelutil
from labm8 import test

FLAGS = app.FLAGS

# The verbatim string printed to stdout by `oclgrind --version`.
VERSION = """
Oclgrind 18.3

Copyright (c) 2013-2018
James Price and Simon McIntosh-Smith, University of Bristol
https://github.com/jrprice/Oclgrind

"""

# Path to helper binary.
OCLGRIND_WORKING_BIN = bazelutil.DataPath(
    'phd/gpu/oclgrind/test/data/oclgrind_working')


def test_Exec_version():
  """Test that the version of oclgrind is as expected."""
  proc = oclgrind.Exec(['--version'])
  # This test will of course fail if the @oclgrind package is updated.
  assert proc.stdout == VERSION
  assert proc.returncode == 0


def test_Exec_opencl_working_app():
  """Run a binary which checks for oclgrind device availability."""
  proc = oclgrind.Exec([str(OCLGRIND_WORKING_BIN)])
  assert 'done' in proc.stderr
  assert proc.returncode == 0


if __name__ == '__main__':
  test.Main()
