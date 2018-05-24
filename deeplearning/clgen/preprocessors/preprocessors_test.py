"""Unit tests for //deeplearning/clgen/preprocessors/preprocessors.py."""
import sys

import pytest
from absl import app
from absl import logging

from deeplearning.clgen import errors
from deeplearning.clgen import languages
from deeplearning.clgen.preprocessors import preprocessors


@preprocessors.clgen_preprocessor
def MockPreprocessor(text: str) -> str:
  """A mock preprocessor."""
  return text


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


# PreprocessDatabase() tests.

def test_PreprocessDatabase_empty(empty_db_path):
  """Test PreprocessDatabase on an empty database."""
  preprocessors.PreprocessDatabase(empty_db_path, languages.Language.OPENCL, [
    'deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor'])


# Benchmarks.

def test_benchmark_GetPreprocessFunction_mock(benchmark):
  """Benchmark GetPreprocessFunction()"""
  benchmark(preprocessors.GetPreprocessorFunction,
            'deeplearning.clgen.preprocessors.preprocessors_test'
            ':MockPreprocessor')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  logging.set_verbosity(logging.DEBUG)
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
