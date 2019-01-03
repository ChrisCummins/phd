"""Unit tests for //deeplearning/clgen/preprocessors/public.py."""

import pytest
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import public
from labm8 import test


FLAGS = flags.FLAGS


def test_clgen_preprocessor_good():
  """Test clgen_preprocessor decorator on a valid function."""

  @public.clgen_preprocessor
  def MockPreprocessor(text: str) -> str:
    """Mock preprocessor."""
    return text

  assert MockPreprocessor('foo') == 'foo'


def test_clgen_preprocessor_missing_return_type():
  """Test clgen_preprocessor on a function missing a return type hint."""
  with pytest.raises(errors.InternalError):
    @public.clgen_preprocessor
    def MockPreprocessor(test: str):
      """Mock preprocessor with a missing return type hint."""
      del test


def test_clgen_preprocessor_missing_argument_type():
  """Test clgen_preprocessor on a function missing an argument type hint."""
  with pytest.raises(errors.InternalError):
    @public.clgen_preprocessor
    def MockPreprocessor(test) -> str:
      """Mock preprocessor with a missing argument type hint."""
      del test


def test_clgen_preprocessor_incorrect_argument_name():
  """Test clgen_preprocessor on a function missing an argument type hint."""
  with pytest.raises(errors.InternalError):
    @public.clgen_preprocessor
    def MockPreprocessor(foo: str) -> str:
      """Mock preprocessor with a mis-named argument."""
      del foo


if __name__ == '__main__':
  test.Main()
