"""Unit tests for //compilers/llvm/opt.py."""

import pytest

from compilers.llvm import opt
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


@pytest.mark.parametrize("o", ("-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"))
def test_ValidateOptimizationLevel_valid(o: str):
  """Test that valid optimization levels are returned."""
  assert opt.ValidateOptimizationLevel(o) == o


@pytest.mark.parametrize(
    "o",
    (
        "O0",  # missing leading '-'
        "-Ofast",  # valid for clang, not for opt
        "-O4",  # not a real value
        "foo"))  # not a real value
def test_ValidateOptimizationLevel_invalid(o: str):
  """Test that invalid optimization levels raise an error."""
  with pytest.raises(ValueError) as e_ctx:
    opt.ValidateOptimizationLevel(o)
  assert o in str(e_ctx.value)


if __name__ == '__main__':
  test.Main()
