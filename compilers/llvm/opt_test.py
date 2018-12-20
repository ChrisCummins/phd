"""Unit tests for //compilers/llvm/opt.py."""
import sys
import typing

import pytest
from absl import app
from absl import flags

from compilers.llvm import opt


FLAGS = flags.FLAGS


@pytest.mark.parametrize(
    "opt", ("-O0", "-O1", "-O2", "-O3", "-Ofast", "-Os", "-Oz"))
def test_ValidateOptimizationLevel_valid(o: str):
  """Test that valid optimization levels are returned."""
  assert opt.ValidateOptimizationLevel(o) == o


@pytest.mark.parametrize(
    "opt", ("O0",  # missing leading '-'
            "-O4",  # not a real value
            "foo"))  # not a real value
def test_ValidateOptimizationLevel_invalid(o: str):
  """Test that invalid optimization levels raise an error."""
  with pytest.raises(ValueError) as e_ctx:
    opt.ValidateOptimizationLevel(o)
  assert o in str(e_ctx.value)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
