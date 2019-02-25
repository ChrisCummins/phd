"""End-to-end compilation pipeline."""
import pathlib
import subprocess

from absl import flags

from compilers.llvm import clang
from compilers.llvm import opt
from labm8 import test

FLAGS = flags.FLAGS


def test_LlvmPipeline(tempdir: pathlib.Path):
  """End-to-end test."""
  with open(tempdir / 'foo.c', 'w') as f:
    f.write("""
#include <stdio.h>

int main() {
  int x = 0;
  if (x != 0)
    x = 5; // dead code
  printf("Hello, world!\\n");
  return x;
}
""")

  # Generate bytecode.
  p = clang.Exec([
      str(tempdir / 'foo.c'), '-o',
      str(tempdir / 'foo.ll'), '-S', '-xc++', '-emit-llvm', '-c', '-O0'
  ])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo.ll').is_file()

  # Run an optimization pass.
  p = opt.Exec(
      [str(tempdir / 'foo.ll'), '-o',
       str(tempdir / 'foo2.ll'), '-S', '-dce'])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo2.ll').is_file()

  # Compile bytecode to LLVM IR.
  p = clang.Exec([str(tempdir / 'foo2.ll'), '-o', str(tempdir / 'foo')])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo').is_file()

  out = subprocess.check_output([str(tempdir / 'foo')], universal_newlines=True)
  assert out == 'Hello, world!\n'


if __name__ == '__main__':
  test.Main()
