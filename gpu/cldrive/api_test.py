# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //gpu/cldrive:api."""
import pytest
import numpy as np
import subprocess

from gpu.cldrive import api
from gpu.cldrive.legacy import env
from gpu.cldrive.proto import cldrive_pb2
from gpu.clinfo.proto import clinfo_pb2
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


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


def test_DriveToDataFrame_columns(device: clinfo_pb2.OpenClDevice):
  df = api.DriveToDataFrame(_MakeInstance(device, ""))
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


def test_DriveToDataFrame_single_program(device: clinfo_pb2.OpenClDevice):
  df = api.DriveToDataFrame(
      _MakeInstance(
          device, """
kernel void A(global int* a) {
  a[get_global_id(0)] = a[get_global_id(0)] * 2;
}
"""))
  assert len(df) == 3  # 3 runs to validate behaviour
  for i in range(3):
    row = df.iloc[i]
    assert row.instance == 0
    assert row.device == device.name
    assert row.build_opts == ""
    assert row.kernel == "A"
    assert row.work_item_local_mem_size == 0
    assert row.work_item_private_mem_size == 0
    assert row.global_size == 1
    assert row.local_size == 1
    assert row.outcome == 'PASS'
    assert row.transferred_bytes == 8
    assert row.runtime_ms > 0


def test_DriveToDataFrame_invalid_program(device: env.OpenCLEnvironment):
  df = api.DriveToDataFrame(_MakeInstance(device, "Invalid program source"))
  assert len(df) == 1
  row = df.iloc[0]
  assert row.instance == 0
  assert row.device == device.name
  assert row.build_opts == ""
  assert row.kernel == ""
  assert np.isnan(row.work_item_local_mem_size)
  assert np.isnan(row.work_item_private_mem_size)
  assert np.isnan(row.global_size)
  assert np.isnan(row.local_size)
  assert row.outcome == 'PROGRAM_COMPILATION_FAILURE'
  assert np.isnan(row.transferred_bytes)
  assert np.isnan(row.runtime_ms)


def test_DriveToDataFrame_no_outputs(device: env.OpenCLEnvironment):
  df = api.DriveToDataFrame(
      _MakeInstance(device, "kernel void A(global int* a) {}"))
  assert len(df) == 1
  row = df.iloc[0]
  assert row.instance == 0
  assert row.device == device.name
  assert row.build_opts == ""
  assert row.kernel == "A"
  assert row.work_item_local_mem_size == 0
  assert row.work_item_private_mem_size == 0
  assert row.global_size == 1
  assert row.local_size == 1
  assert row.outcome == 'NO_OUTPUT'
  assert np.isnan(row.transferred_bytes)
  assert np.isnan(row.runtime_ms)


def test_DriveToDataFrame_input_insensitive(device: env.OpenCLEnvironment):
  df = api.DriveToDataFrame(
      _MakeInstance(
          device,
          "kernel void Foo(global int* a) { a[get_global_id(0)] = 0; }"))
  assert len(df) == 1
  row = df.iloc[0]
  assert row.instance == 0
  assert row.device == device.name
  assert row.build_opts == ""
  assert row.kernel == "Foo"
  assert row.work_item_local_mem_size == 0
  assert row.work_item_private_mem_size == 0
  assert row.global_size == 1
  assert row.local_size == 1
  assert row.outcome == 'INPUT_INSENSITIVE'
  assert np.isnan(row.transferred_bytes)
  assert np.isnan(row.runtime_ms)


def test_DriveToDataFrame_device_not_found(device: env.OpenCLEnvironment):
  device.name = "nope"
  device.platform_name = "not a real platform"
  device.device_name = "not a real device"
  with pytest.raises(subprocess.CalledProcessError):
    api.DriveToDataFrame(
        _MakeInstance(
            device,
            "kernel void A(global int* a) { a[get_global_id(0)] *= 2; }"))


if __name__ == '__main__':
  test.Main()
