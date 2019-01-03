"""Unit tests for //experimental/compilers/random_opt/implementation.py."""
import pathlib

from absl import flags

from compilers.llvm import clang
from experimental.compilers.random_opt import implementation
from labm8 import test


FLAGS = flags.FLAGS


def test_BytecodesAreEqual(tempdir: pathlib.Path):
  """Test binary difftesting."""
  src = tempdir / 'a.c'
  a, b = tempdir / 'a', tempdir / 'b'
  a_opt, b_opt = tempdir / 'a_opt', tempdir / 'b_opt'
  with open(src, 'w') as f:
    f.write("""
int DoFoo(int x) {
  return 2 * x;
}

int main(int argc, char** argv) {
  return DoFoo(10);
}
""")

  clang.Exec([str(src), '-o', str(a), '-O0', '-S', '-c', '-emit-llvm'])
  clang.Exec([str(src), '-o', str(a_opt), '-O3', '-S', '-c', '-emit-llvm'])
  clang.Exec([str(src), '-o', str(b), '-O0', '-S', '-c', '-emit-llvm'])
  clang.Exec([str(src), '-o', str(b_opt), '-O3', '-S', '-c', '-emit-llvm'])

  # FIXME(cec): Remove debugging printout.
  with open(a) as f:
    a_src = f.read()
  print(a_src)

  assert implementation.BytecodesAreEqual(a, b)
  assert not implementation.BytecodesAreEqual(a, a_opt)
  assert implementation.BytecodesAreEqual(a_opt, b_opt)


def test_BinariesAreEqual(tempdir: pathlib.Path):
  """Test binary difftesting."""
  src = tempdir / 'a.c'
  a, b = tempdir / 'a', tempdir / 'b'
  a_opt, b_opt = tempdir / 'a_opt', tempdir / 'b_opt'
  with open(src, 'w') as f:
    f.write("""
int DoFoo(int x) {
  return 2 * x;
}

int main(int argc, char** argv) {
  return DoFoo(10);
}
""")

  clang.Exec([str(src), '-o', str(a), '-O0'])
  clang.Exec([str(src), '-o', str(a_opt), '-O3'])
  clang.Exec([str(src), '-o', str(b), '-O0'])
  clang.Exec([str(src), '-o', str(b_opt), '-O3'])

  assert implementation.BinariesAreEqual(a, b)
  assert not implementation.BinariesAreEqual(a, a_opt)
  assert implementation.BinariesAreEqual(a_opt, b_opt)


if __name__ == '__main__':
  test.Main()
