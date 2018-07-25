"""Unit tests for //gpu/cldrive/env.py."""
import pytest
import sys
from absl import app

from gpu.cldrive import env


def test_OclgrindOpenCLEnvironment_Exec_version():
  """Test that OclgrindOpenCLEnvironment.Exec() works as expected."""
  proc = env.OclgrindOpenCLEnvironment().Exec(['--version'])
  # This test will of course fail if the @oclgrind package is updated.
  assert proc.stdout == """
Oclgrind 18.3

Copyright (c) 2013-2018
James Price and Simon McIntosh-Smith, University of Bristol
https://github.com/jrprice/Oclgrind

"""


def test_OclgrindOpenCLEnvironment_name():
  """Test that the OclgrindOpenCLEnvironment has a correct 'name' property."""
  env_ = env.OclgrindOpenCLEnvironment()
  # This test will of course fail if the @oclgrind package is updated.
  assert 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2' == env_.name


def main(argv):  # pylint: disable=missing-docstring
  """Main entry point."""
  del argv
  sys.exit(pytest.main(
      [env.__file__, __file__, "-vv", "--doctest-modules"]))


if __name__ == "__main__":
  app.run(main)
