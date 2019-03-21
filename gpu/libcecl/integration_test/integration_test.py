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
import pathlib

import pytest

from compilers.llvm import clang
from gpu.cldrive.legacy import env as cldrive_env
from gpu.libcecl import libcecl_compile
from gpu.libcecl import libcecl_rewriter
from gpu.libcecl import libcecl_runtime
from gpu.oclgrind import oclgrind
from labm8 import app
from labm8 import bazelutil
from labm8 import test

FLAGS = app.FLAGS

_HELLO = bazelutil.DataPath('phd/gpu/libcecl/integration_test/hello.cc')


@pytest.fixture(scope='function')
def hello_src() -> str:
  """Test fixture which returns the 'hello world' OpenCL app source."""
  with open(_HELLO) as f:
    return f.read()


def test_rewrite_compile_link_execute(tempdir: pathlib.Path, hello_src: str):
  """Test end-to-end libcecl pipeline."""
  # Re-write OpenCL source to use libcecl.
  libcecl_src = libcecl_rewriter.RewriteOpenClSource(hello_src)

  # Compile libcecl source to bytecode.
  bytecode_path = tempdir / 'a.ll'
  cflags, ldflags = libcecl_compile.LibCeclCompileAndLinkFlags()

  proc = clang.Exec(
      ['-x', 'c++', '-', '-S', '-emit-llvm', '-o',
       str(bytecode_path)] + cflags,
      stdin=libcecl_src,
      stdout=None,
      stderr=None)
  assert not proc.returncode
  assert bytecode_path.is_file()

  # Compile bytecode to executable annd link.
  bin_path = tempdir / 'a.out'
  proc = clang.Exec(
      ['-o', str(bin_path), str(bytecode_path)] + ldflags,
      stdout=None,
      stderr=None)
  assert not proc.returncode
  assert bin_path.is_file()

  # Run executable on oclgrind.
  log = libcecl_runtime.RunLibceclExecutable(
      [oclgrind.OCLGRIND_PATH, bin_path],
      cldrive_env.OclgrindOpenCLEnvironment())

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
