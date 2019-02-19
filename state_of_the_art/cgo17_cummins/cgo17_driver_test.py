"""Unit tests for //gpu/cldrive:cgo17_driver."""
import pytest
from absl import flags

from gpu.cldrive import cgo17_driver
from gpu.cldrive import env as cldrive_env
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def opencl_env() -> cldrive_env.OpenCLEnvironment:
  return cldrive_env.OclgrindOpenCLEnvironment()


def test_Drive_log_count(opencl_env: cldrive_env.OpenCLEnvironment):
  """Test that the correct number of logs are retunred."""
  logs = cgo17_driver.Drive("""
kernel void A(global int* a, global int* b) {
  a[get_global_id(0)] += b[get_global_id(0)];
}
""", 128, 128, opencl_env, 5)
  assert len(logs) == 5


if __name__ == '__main__':
  test.Main()
