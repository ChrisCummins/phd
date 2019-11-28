"""Unit tests for //datasets/benchmarks/jacobi_opencl."""
from datasets.benchmarks.jacobi_opencl import jacobi_opencl
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_GetDeviceList():
  # Warning: This test will fail if no OpenCL device is available.
  assert jacobi_opencl.GetDeviceList()


def test_RunJacobiBenchmark():
  # Warning: This test will fail if no OpenCL device is available.
  device = jacobi_opencl.GetDeviceList()[0]
  config = {
      "norder": 128,
      "iteration_count": 10,
      "datatype": "float",
      "convergence_frequency": 0,
      "convergence_tolerance": 0.001,
      "wgsize": [32, 1],
      "unroll": 1,
      "layout": "col-major",
      "conditional": "branch",
      "fmad": "op",
      "divide_A": "normal",
      "addrspace_b": "global",
      "addrspace_xold": "global",
      "integer": "int",
      "relaxed_math": False,
      "use_const": False,
      "use_restrict": False,
      "use_mad24": False,
      "const_norder": False,
      "const_wgsize": False,
      "coalesce_cols": True,
      "min_runtime": 0,
      "max_runtime": 0
  }
  jacobi_opencl.RunJacobiBenchmark(config, device)


if __name__ == '__main__':
  test.Main()
