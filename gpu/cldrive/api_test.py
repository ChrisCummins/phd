"""Unit tests for //gpu/cldrive:api."""
import pytest
from absl import flags

from gpu.cldrive import api
from gpu.cldrive.legacy import env
from gpu.cldrive.proto import cldrive_pb2
from gpu.clinfo.proto import clinfo_pb2
from labm8 import test

FLAGS = flags.FLAGS


def _MakeInstance(device, src, num_runs: int = 1, dynamic_params=[
    (1, 1),
]):
  """Utility function to generate an instances proto."""
  return cldrive_pb2.CldriveInstances(instance=[
      cldrive_pb2.CldriveInstance(
          device=device,
          opencl_src=src,
          min_runs_per_kernel=num_runs,
          dynamic_params=[
              cldrive_pb2.DynamicParams(global_size_x=g, local_size_x=l)
              for g, l in dynamic_params
          ]),
  ])


@pytest.fixture(scope='session')
def device() -> env.OpenCLEnvironment:
  """Test fixture which yields a testing OpenCL device."""
  return env.OclgrindOpenCLEnvironment().proto


@pytest.fixture(scope='function')
def input1(device: clinfo_pb2.OpenClDevice) -> cldrive_pb2.CldriveInstances:
  """Test fixture that returns a very simple kernel."""
  return _MakeInstance(
      device, """
kernel void A(global int* a) {
  a[get_global_id(0)] = a[get_global_id(0)] * 2;
}
""")


def test_Drive_smoke_test(input1: cldrive_pb2.CldriveInstances):
  """Test that Drive doesn't blow up."""
  api.Drive(input1)


def test_DriveToDataFrame_columns(input1: cldrive_pb2.CldriveInstances):
  df = api.DriveToDataFrame(input1)
  assert list(df.columns.values) == [
      'instance',
      'device',
      'build_opts',
      'kernel',
      'work_item_local_mem_size',
      'work_item_private_mem_size',
      'global_size',
      'local_size',
      'outcome',
      'transferred_bytes',
      'runtime_ms',
  ]


def test_DriveToDataFrame_num_rows(input1: cldrive_pb2.CldriveInstances):
  df = api.DriveToDataFrame(input1)
  assert len(df) == 3  # 3 runs to validate behaviour


def test_DriveToDataFrame_instance(input1: cldrive_pb2.CldriveInstances):
  df = api.DriveToDataFrame(input1)
  assert set(df['instance'].values) == {0}


def test_DriveToDataFrame_instance(input1: cldrive_pb2.CldriveInstances):
  df = api.DriveToDataFrame(input1)
  assert set(df['build_opts'].values) == {''}


def test_DriveToDataFrame_kernel(input1: cldrive_pb2.CldriveInstances):
  df = api.DriveToDataFrame(input1)
  assert set(df['kernel'].values) == {"A"}


@pytest.fixture(scope='function')
def input2(device: clinfo_pb2.OpenClDevice) -> cldrive_pb2.CldriveInstances:
  """Test fixture that returns a very simple kernel."""
  return _MakeInstance(device, "Invalid OpenCL program source")


def test_DriveToDataFrame_num_rows_on_error(
    input2: cldrive_pb2.CldriveInstances):
  df = api.DriveToDataFrame(input2)
  assert len(df) == 1


if __name__ == '__main__':
  test.Main()
