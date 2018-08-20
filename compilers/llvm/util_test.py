"""Unit tests for //compilers/llvm/util.py."""
import pytest
import sys
import typing
from absl import app
from absl import flags

from compilers.llvm import llvm
from compilers.llvm import util


FLAGS = flags.FLAGS


@pytest.mark.parametrize('cflags', [['-O0'], ['-O1'], ['-O2'], ['-O3']])
def test_GetOptArgs_black_box(cflags: typing.List[str]):
  """Black box opt args test."""
  args = util.GetOptArgs(cflags)
  assert args
  for invocation in args:
    assert invocation


def test_GetOptArgs_bad_args():
  """Error is raised if invalid args are passed."""
  with pytest.raises(llvm.LlvmError):
    util.GetOptArgs(['-not-a-real-arg!'])


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
