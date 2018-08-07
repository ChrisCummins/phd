"""Unit tests for //compilers/llvm/clang.py."""
import pathlib
import pytest
import sys
import tempfile
import typing
from absl import app
from absl import flags

from compilers.llvm import clang


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory() as d:
    yield pathlib.Path(d)


def test_Exec_compile_bytecode(tempdir: pathlib.Path):
  """Test bytecode generation."""
  with open(tempdir / 'foo.cc', 'w') as f:
    f.write("""
#include <iostream>

int main() {
  std::cout << "Hello, world!" << std::endl;
  return 0;
}
""")
  p = clang.Exec([str(tempdir / 'foo.cc'), '-xc++', '-S', '-emit-llvm', '-c',
                  '-o', str(tempdir / 'foo.ll')])
  assert not p.returncode
  assert not p.stderr
  assert not p.stdout
  assert (tempdir / 'foo.ll').is_file()


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
