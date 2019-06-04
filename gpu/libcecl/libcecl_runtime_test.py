"""Unit tests for //gpu/libcecl:libcecl_runtime."""
import pytest

from gpu.cldrive.legacy import env as cldrive_env
from gpu.clinfo.proto import clinfo_pb2
from gpu.libcecl import libcecl_runtime
from labm8 import test


def test_KernelInvocationsFromCeclLog():
  device = cldrive_env.OpenCLEnvironment(
      clinfo_pb2.OpenClDevice(
          device_type='CPU',
          device_name='OpenCL Device',
      ))
  invocations = libcecl_runtime.KernelInvocationsFromCeclLog(
      [
          'clCreateProgramWithSource',
          'BEGIN PROGRAM SOURCE',
          'kernel void Kernel(global int* a) {}',
          'END PROGRAM SOURCE',
          'clCreateCommandQueue ; CPU ; OpenCL Device',
          'clBuildProgram ; Kernel'
          'clEnqueueMapBuffer ; Buffer ; 1024 ; 2000',
          'clEnqueueNDRangeKernel ; Kernel ; 128 ; 64 ; 3000',
          'clEnqueueTask ; Kernel ; 4000',
      ],
      expected_device_name=device.device_name,
      expected_devtype=device.device_type)
  assert len(invocations) == 2


def test_KernelInvocationsFromCeclLog_different_device_type():
  with pytest.raises(ValueError):
    libcecl_runtime.KernelInvocationsFromCeclLog(
        [
            'clCreateCommandQueue ; GPU ; OpenCL Device',
        ],
        expected_devtype='CPU',
        expected_device_name='OpenCL Device')


def test_KernelInvocationsFromCeclLog_different_device_name():
  with pytest.raises(ValueError):
    libcecl_runtime.KernelInvocationsFromCeclLog(
        [
            'clCreateCommandQueue ; CPU ; Different OpenCL Device',
        ],
        expected_devtype='CPU',
        expected_device_name='OpenCL Device')


if __name__ == '__main__':
  test.Main()
