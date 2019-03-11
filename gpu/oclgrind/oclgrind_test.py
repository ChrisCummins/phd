"""Unit tests for //gpu/oclgrind/oclgrind.py."""

from gpu.oclgrind import oclgrind
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

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


if __name__ == '__main__':
  test.Main()
