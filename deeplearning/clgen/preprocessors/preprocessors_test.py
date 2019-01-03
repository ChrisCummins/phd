"""Unit tests for //deeplearning/clgen/preprocessors/preprocessors.py."""

import pytest
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.preprocessors import public
from labm8 import test


FLAGS = flags.FLAGS


@public.clgen_preprocessor
def MockPreprocessor(text: str) -> str:
  """A mock preprocessor."""
  del text
  return 'PREPROCESSED'


@public.clgen_preprocessor
def MockPreprocessorBadCode(text: str) -> str:
  """A mock preprocessor which raises a BadCodeException."""
  del text
  raise errors.BadCodeException('bad code')


@public.clgen_preprocessor
def MockPreprocessorInternalError(text: str) -> str:
  """A mock preprocessor which raises a BadCodeException."""
  del text
  raise errors.InternalError('internal error')


def MockUndecoratedPreprocessor(text: str) -> str:
  """A mock preprocessor which is not decorated with @clgen_preprocessor."""
  return text


# GetPreprocessFunction() tests.

def test_GetPreprocessFunction_empty_string():
  """Test that an UserError is raised if no preprocessor is given."""
  with pytest.raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction('')
  assert 'Invalid preprocessor name' in str(e_info.value)


def test_GetPreprocessFunction_missing_module():
  """Test that UserError is raised if module not found."""
  with pytest.raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction('not.a.real.module:Foo')
  assert 'not found' in str(e_info.value)


def test_GetPreprocessFunction_missing_function():
  """Test that UserError is raised if module exists but function doesn't."""
  with pytest.raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction(
        'deeplearning.clgen.preprocessors.preprocessors_test:Foo')
  assert 'not found' in str(e_info.value)


def test_GetPreprocessFunction_undecorated_preprocessor():
  """Test that an UserError is raised if preprocessor not decorated."""
  with pytest.raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction(
        'deeplearning.clgen.preprocessors.preprocessors_test'
        ':MockUndecoratedPreprocessor')
  assert '@clgen_preprocessor' in str(e_info.value)


def test_GetPreprocessFunction_mock_preprocessor():
  """Test that a mock preprocessor can be found."""
  f = preprocessors.GetPreprocessorFunction(
      'deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor')
  assert f == MockPreprocessor


# Preprocess() tests.


def test_Preprocess_no_preprocessors():
  """Test unmodified output if no preprocessors."""
  assert preprocessors.Preprocess('hello', []) == 'hello'


def test_Preprocess_mock_preprocessor():
  """Test unmodified output if no preprocessors."""
  assert preprocessors.Preprocess('hello', [
    'deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor']) \
         == 'PREPROCESSED'


def test_Preprocess_mock_preprocessor_bad_code():
  """Test that BadCodeException is propagated."""
  with pytest.raises(errors.BadCodeException):
    preprocessors.Preprocess('', [
      'deeplearning.clgen.preprocessors.preprocessors_test'
      ':MockPreprocessorBadCode'])


def test_Preprocess_mock_preprocessor_internal_error():
  """Test that InternalError is propagated."""
  with pytest.raises(errors.InternalError):
    preprocessors.Preprocess('', [
      'deeplearning.clgen.preprocessors.preprocessors_test'
      ':MockPreprocessorInternalError'])


# Benchmarks.

def test_benchmark_GetPreprocessFunction_mock(benchmark):
  """Benchmark GetPreprocessFunction."""
  benchmark(preprocessors.GetPreprocessorFunction,
            'deeplearning.clgen.preprocessors.preprocessors_test'
            ':MockPreprocessor')


if __name__ == '__main__':
  test.Main()
