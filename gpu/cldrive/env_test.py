"""Unit tests for //gpu/cldrive/env.py."""

from absl import flags

from gpu.cldrive import env
from labm8 import test


FLAGS = flags.FLAGS


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


if __name__ == "__main__":
  test.Main()
