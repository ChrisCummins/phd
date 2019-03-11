"""Unit tests for //compilers/llvm/util.py."""
import typing

import pytest

from compilers.llvm import llvm
from compilers.llvm import util
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


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


if __name__ == '__main__':
  test.Main()
