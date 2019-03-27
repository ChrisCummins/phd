"""Unit tests for //gpu/libcecl:libcecl_compile."""
import pytest
import pathlib
import subprocess

from compilers.llvm import clang
from gpu.libcecl import libcecl_compile
from labm8 import test


@pytest.fixture(scope='function')
def hello_world_c_src() -> str:
  return """
#include <stdio.h>

int main(int argc, char** argv) {
  printf("Hello, world!\\n");
  return 0;
}
"""


@pytest.mark.parametrize('flags_getter',
                         (libcecl_compile.OpenClCompileAndLinkFlags,
                          libcecl_compile.LibCeclCompileAndLinkFlags))
@pytest.mark.parametrize('flags_getter_args', ({
    "opencl_headers": True
}, {
    "opencl_headers": False
}))
def test_OpenClCompileAndLinkFlags_smoke_test(flags_getter, flags_getter_args,
                                              hello_world_c_src: str,
                                              tempdir: pathlib.Path):
  """Test that code can be compiled with OpenCL flags."""
  cflags, ldflags = flags_getter(**flags_getter_args)

  # Create bytecode.
  bytecode_path = tempdir / 'a.ll'
  proc = clang.Exec(
      ['-x', 'c', '-', '-S', '-emit-llvm', '-o',
       str(bytecode_path)] + cflags,
      stdin=hello_world_c_src,
      stdout=None,
      stderr=None)
  assert not proc.returncode
  assert bytecode_path.is_file()

  # Create executable.
  bin_path = tempdir / 'a.out'
  proc = clang.Exec(
      ['-o', str(bin_path), str(bytecode_path)] + ldflags,
      stdout=None,
      stderr=None)
  assert not proc.returncode
  assert bin_path.is_file()

  assert subprocess.check_output(
      [str(bin_path)],
      universal_newlines=True,
      env=libcecl_compile.LibCeclExecutableEnvironmentVariables(
      )) == 'Hello, world!\n'


if __name__ == '__main__':
  test.Main()
