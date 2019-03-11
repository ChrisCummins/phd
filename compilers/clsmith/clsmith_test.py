"""Unit tests for //compilers/clsmith/clsmith.py."""

import pytest

from compilers.clsmith import clsmith
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_Exec_no_args():
  """Test that CLSmith returns a generated file."""
  src = clsmith.Exec()
  # Check the basic structure of the generated file.
  assert src.startswith('// -g ')
  assert '__kernel void ' in src


def test_Exec_invalid_argument():
  """Test that CLSmithError is raised if invalid args passed to CLSmith."""
  with pytest.raises(clsmith.CLSmithError) as e_ctx:
    clsmith.Exec('--invalid_opt')
  assert '' == str(e_ctx.value)
  assert e_ctx.value.returncode == 255


if __name__ == '__main__':
  test.Main()
