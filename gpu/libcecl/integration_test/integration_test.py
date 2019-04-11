# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of libcecl.
#
# libcecl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libcecl.  If not, see <https://www.gnu.org/licenses/>.
"""Integration test for //gpu/libcecl."""
import io
import pathlib
import subprocess
import re

import pytest

from compilers.llvm import clang
from gpu.cldrive.legacy import env as cldrive_env
from gpu.libcecl import libcecl_compile
from gpu.libcecl.proto import libcecl_pb2
from gpu.libcecl import libcecl_rewriter
from gpu.libcecl import libcecl_runtime
from gpu.oclgrind import oclgrind
from labm8 import app
from labm8 import bazelutil
from labm8 import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = 'gpu'

_CLINFO = bazelutil.DataPath('phd/third_party/clinfo/clinfo.c')
_HELLO = bazelutil.DataPath('phd/gpu/libcecl/integration_test/hello.cc')


@pytest.fixture(scope='function')
def hello_src() -> str:
  """Test fixture which returns the 'hello world' OpenCL app source."""
  with open(_HELLO) as f:
    return f.read()


@pytest.fixture(scope='function')
def clinfo_src() -> str:
  """Test fixture which returns the C source code for a clinfo program."""
  with open(_CLINFO, 'rb') as f:
    return f.read().decode('utf-8').encode('ascii', 'ignore').decode('ascii')


def _RewriteCompileLinkExecute(
    outdir: pathlib.Path,
    src: str,
    lang: str = 'c++',
    extra_ldflags=None,
    extra_cflags=None,
    extra_exec_args=None) -> libcecl_pb2.LibceclExecutableRun:
  """Compile, link, and execute a program using libcecl."""
  # Re-write OpenCL source to use libcecl.
  # libcecl_src = libcecl_rewriter.RewriteOpenClSource(src)
  libcecl_src = src

  # Compile libcecl source to bytecode.
  src_path = outdir / f'a.txt'
  objectfile_path = outdir / 'a.o'
  cflags, ldflags = libcecl_compile.LibCeclCompileAndLinkFlags()

  with open(src_path, 'w') as f:
    f.write(libcecl_src)
  extra_cflags = extra_cflags or []
  subprocess.check_call(
      ['clang++', '-x', lang,
       str(src_path), '-c', '-o',
       str(objectfile_path)] + cflags + extra_cflags)
  assert objectfile_path.is_file()

  # Compile bytecode to executable and link.
  bin_path = outdir / 'a.out'
  extra_ldflags = extra_ldflags or []
  subprocess.check_call(
      ['clang++', '-o', str(bin_path),
       str(objectfile_path)] + ldflags + extra_ldflags)
  assert bin_path.is_file()

  # Run executable on oclgrind.
  extra_exec_args = extra_exec_args or []
  return libcecl_runtime.RunLibceclExecutable(
      [str(bin_path)] + extra_exec_args,
      cldrive_env.OclgrindOpenCLEnvironment())


def test_rewrite_compile_link_execute_clinfo(tempdir: pathlib.Path,
                                             clinfo_src: str):
  log = _RewriteCompileLinkExecute(
      tempdir,
      clinfo_src,
      lang='c',
      extra_ldflags=['-lm', '-lstdc++'],
      extra_exec_args=['--raw'])

  assert log.ms_since_unix_epoch
  assert log.returncode == 0
  assert log.device == cldrive_env.OclgrindOpenCLEnvironment().proto
  assert len(log.kernel_invocation) == 0
  assert len(log.opencl_program_source) == 0

  assert not log.stderr
  assert re.match(
      r"0 CL_PLATFORM_NAME Oclgrind\n"
      r"0 CL_PLATFORM_VERSION OpenCL \d+\.\d+ \(Oclgrind [\d\.]+\)\n"
      r"0:0 CL_DEVICE_NAME Oclgrind Simulator\n"
      r"0:0 CL_DEVICE_TYPE [a-zA-Z |]+\n"
      r"0:0 CL_DEVICE_VERSION OpenCL \d+\.\d+ \(Oclgrind [\d\.]+\)\n"
      r"0:0 CL_DEVICE_GLOBAL_MEM_SIZE \d+\n"
      r"0:0 CL_DEVICE_LOCAL_MEM_SIZE \d+\n"
      r"0:0 CL_DEVICE_MAX_WORK_GROUP_SIZE \d+\n"
      r"0:0 CL_DEVICE_MAX_WORK_ITEM_SIZES \(\d+, \d+, \d+\)\n", log.stdout,
      re.MULTILINE)


def test_rewrite_compile_link_execute(tempdir: pathlib.Path, hello_src: str):
  """Test end-to-end libcecl pipeline."""
  log = _RewriteCompileLinkExecute(
      tempdir, hello_src, lang='c++', extra_cflags=['-std=c++11'])

  print(log.stdout)
  print(log.stderr)

  # Check values in log.
  assert log.ms_since_unix_epoch
  assert log.returncode == 0
  assert log.device == cldrive_env.OclgrindOpenCLEnvironment().proto
  assert len(log.kernel_invocation) == 1
  assert len(log.opencl_program_source) == 1
  assert log.opencl_program_source[0] == """\
kernel void square(
    global float* input,
    global float* output,
    const unsigned int count) {
  int i = get_global_id(0);
  if(i < count)
    output[i] = input[i] * input[i];
}"""

  assert log.kernel_invocation[0].kernel_name == 'square'
  assert log.kernel_invocation[0].global_size == 1024
  assert log.kernel_invocation[0].local_size == 1024
  assert log.kernel_invocation[0].transferred_bytes == 8192
  assert log.kernel_invocation[
      0].transfer_time_ns > 1000  # Flaky, but probably true.
  assert log.kernel_invocation[
      0].kernel_time_ns > 1000  # Flaky, but probably true.

  profile_time = (log.kernel_invocation[0].transfer_time_ns +
                  log.kernel_invocation[0].kernel_time_ns)
  assert 1000 < log.elapsed_time_ns < profile_time


if __name__ == '__main__':
  test.Main()
