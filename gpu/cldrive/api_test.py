"""Unit tests for //gpu/cldrive:api."""
import pytest
from absl import flags

from gpu.cldrive import api
from gpu.cldrive.legacy import env
from gpu.cldrive.proto import cldrive_pb2
from gpu.clinfo.proto import clinfo_pb2
from labm8 import test

FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def device() -> env.OpenCLEnvironment:
  """Test fixture which yields a testing OpenCL device."""
  return env.OclgrindOpenCLEnvironment().proto


def test_Drive_num_kernels(device: clinfo_pb2.OpenClDevice):
  """Test that one kernel is found in source."""
  instance = cldrive_pb2.CldriveInstance(
      device=device,
      opencl_src="kernel void A() {}",
      min_runs_per_kernel=1,
      dynamic_params=[
          cldrive_pb2.DynamicParams(global_size_x=1, local_size_x=1)
      ])
  api.Drive(instance)
  assert len(instance.kernel) == 1


if __name__ == '__main__':
  test.Main()
