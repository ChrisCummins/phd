"""This is an implementation of the OpenCL benchmark driver described in:

    ﻿Cummins, C., Petoumenos, P., Zang, W., & Leather, H. (2017). Synthesizing
    Benchmarks for Predictive Modeling. In CGO. IEEE.

Note this is not the same implementation as was used to generate the data for
the paper. For that, see the paper's artifact in //docs/2017_02_cgo/code.
"""
import pathlib
import tempfile
import typing

from compilers.llvm import clang
from deeplearning.deepsmith.harnesses import cldrive as cldrive_harness
from deeplearning.deepsmith.proto import deepsmith_pb2
from gpu.cldrive.legacy import env as cldrive_env
from gpu.cldrive.proto import cldrive_pb2
from gpu.libcecl import libcecl_compile
from gpu.libcecl import libcecl_rewriter
from gpu.libcecl import libcecl_runtime
from gpu.libcecl.proto import libcecl_pb2
from labm8 import app

FLAGS = app.FLAGS

# All the combinations of local and global sizes used for synthetic kernels in
# the CGO'17 experiments. These are the first dimension values, the other two
# dimensions are ones. E.g. the tuple (64, 128) means a local (workgroup) size
# of (64, 1, 1), and a global size of (128, 1, 1).
LSIZE_GSIZE_PAIRS = [
    (64, 64),
    (128, 128),
    (256, 256),
    (256, 512),
    (256, 1024),
    (256, 2048),
    (256, 4096),
    (256, 8192),
    (256, 16384),
    (256, 65536),
    (256, 131072),
    (256, 262144),
    (256, 524288),
    (256, 1048576),
    (256, 2097152),
    (256, 4194304),
]


class DriverFailure(ValueError):
  """Base class for driver failures."""
  pass


class DriverConstructionFailed(DriverFailure):
  pass


class DriverCompilationFailed(DriverConstructionFailed):
  pass


class DriverExecutionFailed(DriverFailure):
  pass


class KernelIsNondeterministic(DriverFailure):

  def __init__(self, output1: str, output2: str):
    self.output1, self.output2 = output1, output2


class KernelIsInputInsenstive(DriverFailure):
  pass


class KernelProducesNoOutput(DriverFailure):
  pass


def Drive(opencl_kernel: str,
          lsize_x: int,
          gsize_x: int,
          opencl_env: cldrive_env.OpenCLEnvironment,
          num_runs: int = 30) -> typing.List[libcecl_pb2.LibceclExecutableRun]:
  """Drive an OpenCL kernel and return the execution logs.

  Args:
    opencl_kernel: The OpenCL kernel to drive, as a string.
    lsize_x: The 1-D local size to use.
    gsize_x: The 1-D global size to use.
    opencl_env: The OpenCL environment to drive the kernel using.
    num_runs: The number of times to execute the kernel.

  Returns:
    A list of LibceclExecutableRun messages.
  """

  cldrive_pb2.CldriveInstance(
      device=cldrive_env.OclgrindOpenCLEnvironment().proto,
      opencl_src="""
kernel void A(global int* a, global float* b, const int c) {
if (get_global_id(0) < c) {
  a[get_global_id(0)] = get_global_id(0);
  b[get_global_id(0)] *= 2.0;
}
}""",
      min_runs_per_kernel=30,
      dynamic_params=[
          cldrive_pb2.DynamicParams(
              global_size_x=16,
              local_size_x=16,
          ),
          cldrive_pb2.DynamicParams(
              global_size_x=1024,
              local_size_x=64,
          ),
          cldrive_pb2.DynamicParams(
              global_size_x=128,
              local_size_x=64,
          ),
      ],
  )

  # Create a pair of driver sources.
  def MakeDriverSrc(data_generator: str) -> str:
    testcase = deepsmith_pb2.Testcase(
        inputs={
            'gsize': f'{gsize_x},1,1',
            'lsize': f'{lsize_x},1,1',
            'data_generator': data_generator,
            'src': opencl_kernel,
        })
    src = cldrive_harness.MakeDriver(testcase, optimizations=True)
    if testcase.invariant_opts['driver_type'] != 'compile_and_run':
      raise DriverConstructionFailed("Expected compile-and-run driver, found " +
                                     testcase.invariant_opts['driver_type'])
    return src

  src1 = MakeDriverSrc('ones')
  src2 = MakeDriverSrc('arange')

  # Re-write OpenCL source to use libcecl.
  libcecl_src1 = libcecl_rewriter.RewriteOpenClSource(src1)
  libcecl_src2 = libcecl_rewriter.RewriteOpenClSource(src2)
  assert libcecl_src1 != libcecl_src2

  with tempfile.TemporaryDirectory(prefix='phd_experimental_') as d:
    tempdir = pathlib.Path(d)

    # Compile libcecl source to binary.
    bin1, bin2 = tempdir / 'a.out', tempdir / 'b.out'
    cflags, ldflags = libcecl_compile.LibCeclCompileAndLinkFlags()

    def CompileDriver(libcecl_src: str, binary_path: pathlib.Path):
      proc = clang.Exec(
          ['-x', 'c', '-std=c99', '-', '-o',
           str(binary_path)] + cflags + ldflags,
          stdin=libcecl_src)
      if proc.returncode:
        raise DriverCompilationFailed(proc.stderr[:1024])
      assert binary_path.is_file()

    CompileDriver(libcecl_src1, bin1)
    CompileDriver(libcecl_src2, bin2)

    def Run(binary_path: pathlib.Path):
      log = libcecl_runtime.RunLibceclExecutable([str(binary_path)], opencl_env)
      if log.returncode:
        raise DriverExecutionFailed(log.stderr[:1024])
      if len(log.kernel_invocation) != 1:
        raise DriverExecutionFailed("Expected 1 OpenCL kernel invocation. "
                                    f"Found {len(log.kernel_invocation)}")
      return log

    def RunTwice(binary_path: pathlib.Path):
      run1 = Run(binary_path)
      run2 = Run(binary_path)
      if run1.stdout != run2.stdout:
        raise KernelIsNondeterministic(run1.stdout, run2.stdout)
      return run1, run2

    def OutputsArentAllOnes(log: libcecl_pb2.LibceclExecutableRun):
      lines = log.stdout.rstrip().split('\n')
      all_ones = set()
      for line in lines:
        arg_name, arg_values = line.split(':')
        all_ones.add(set(arg_values) == {' ', '1'})

      if all_ones == {True}:
        raise KernelProducesNoOutput()

    run1, run2 = RunTwice(bin1)
    OutputsArentAllOnes(run1)
    run3, run4 = RunTwice(bin2)
    if run1.stdout == run3.stdout:
      raise KernelIsInputInsenstive()

    logs = [run1, run2, run3, run4]
    for i in range(4, num_runs):
      logs.append(Run(bin2))

  return logs
