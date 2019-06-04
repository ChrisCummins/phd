"""Unit tests for //gpu/libcecl:libcecl_compile."""
import pathlib
import pytest
import subprocess

from compilers.llvm import clang
from gpu.libcecl import libcecl_compile
from labm8 import test


@pytest.fixture(scope='function')
def c_program_src() -> str:
  """A fixture that returns the code for a simple C program."""
  return """
int main(int argc, char** argv) {
  return 5;
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
def test_OpenClCompileAndLinkFlags_smoke_test(
    flags_getter, flags_getter_args, c_program_src: str, tempdir: pathlib.Path):
  """Test that code can be compiled with OpenCL flags."""
  cflags, ldflags = flags_getter(**flags_getter_args)

  # Create bitcode.
  bitcode_path = tempdir / 'a.ll'
  proc = clang.Exec(
      ['-x', 'c', '-', '-S', '-emit-llvm', '-o',
       str(bitcode_path)] + cflags,
      stdin=c_program_src,
      stdout=None,
      stderr=None)
  assert not proc.returncode
  assert bitcode_path.is_file()

  # Compile bitcode to executable.
  bin_path = tempdir / 'a.out'
  proc = clang.Exec(
      ['-o', str(bin_path), str(bitcode_path)] + ldflags,
      stdout=None,
      stderr=None)
  assert not proc.returncode
  assert bin_path.is_file()

  # The C program should exit with returncode 5.
  proc = subprocess.Popen(
      [str(bin_path)],
      env=libcecl_compile.LibCeclExecutableEnvironmentVariables())
  proc.communicate()
  assert proc.returncode == 5


if __name__ == '__main__':
  test.Main()
