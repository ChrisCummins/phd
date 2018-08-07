"""End-to-end compilation pipeline."""
import pathlib
import pytest
import subprocess
import sys
import tempfile
import typing
from absl import app
from absl import flags

from compilers.llvm import clang
from compilers.llvm import opt


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory() as d:
    yield pathlib.Path(d)


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
  p = clang.Exec([str(tempdir / 'foo.c'), '-o', str(tempdir / 'foo.ll'),
                  '-S', '-xc++', '-emit-llvm', '-c', '-O0'])
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / 'foo.ll').is_file()

  # Run an optimization pass.
  p = opt.Exec([str(tempdir / 'foo.ll'), '-o', str(tempdir / 'foo2.ll'),
                '-S', '-dce'])
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


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
