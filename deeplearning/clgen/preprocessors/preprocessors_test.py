"""Unit tests for //deeplearning/clgen/preprocessors/preprocessors.py."""
import sys

import pytest
from absl import app
from absl import logging

from deeplearning.clgen import dbutil
from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.preprocessors import public


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


def test_Preprocess_mock_preprocessor_bad_code():
  """Test that InternalError is propagated."""
  with pytest.raises(errors.InternalError):
    preprocessors.Preprocess('', [
      'deeplearning.clgen.preprocessors.preprocessors_test'
      ':MockPreprocessorInternalError'])


# PreprocessDatabase() tests.

def test_PreprocessDatabase_empty(empty_db_path):
  """Test PreprocessDatabase on an empty database."""
  preprocessors.PreprocessDatabase(empty_db_path, [
    'deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor'])


def test_PreprocessDatabase_abc(abc_db_path):
  """Test PreprocessDatabase on an empty database."""
  preprocessors.PreprocessDatabase(abc_db_path, [
    'deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor'])
  assert dbutil.num_rows_in(abc_db_path, "PreprocessedFiles") == 3
  db = dbutil.connect(abc_db_path)
  c = db.cursor()
  results = c.execute(
      'SELECT id,status,contents FROM PreprocessedFiles').fetchall()
  assert set([r[0] for r in results]) == {'a', 'b', 'c'}
  assert set([r[1] for r in results]) == {0}
  assert set([r[2] for r in results]) == {'PREPROCESSED', }


def test_PreprocessDatabase_abc_no_preprocessors(abc_db_path):
  """Test that contentfiles are not modified if there's no preprocessors."""
  preprocessors.PreprocessDatabase(abc_db_path, [])
  assert dbutil.num_rows_in(abc_db_path, "PreprocessedFiles") == 3
  db = dbutil.connect(abc_db_path)
  c = db.cursor()
  results = c.execute(
      'SELECT id,status,contents FROM PreprocessedFiles').fetchall()
  assert set([r[0] for r in results]) == {'a', 'b', 'c'}
  assert set([r[1] for r in results]) == {0}
  # The abc_db contentfiles, unmodified.
  assert set([r[2] for r in results]) == {'foo', 'car', 'bar'}


def test_PreprocessDatabase_invalid_preprocessor(abc_db_path):
  """Test that an invalid preprocessor raises an InternalError"""
  with pytest.raises(errors.UserError) as e_info:
    preprocessors.PreprocessDatabase(abc_db_path,
                                     ['not.a.real:Preprocessor'])
  assert 'not.a.real:Preprocessor' in str(e_info.value)
  # Check that nothing has been added to the PreprocessedFiles table.
  db = dbutil.connect(abc_db_path)
  c = db.cursor()
  results = c.execute('SELECT Count(*) FROM PreprocessedFiles').fetchone()
  assert not results[0]


def test_PreprocessDatabase_abc_bad_code(abc_db_path):
  """Test PreprocessDatabase with a bad code preprocessor."""
  preprocessors.PreprocessDatabase(abc_db_path, [
    'deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor',
    'deeplearning.clgen.preprocessors.preprocessors_test'
    ':MockPreprocessorBadCode'])
  assert dbutil.num_rows_in(abc_db_path, "PreprocessedFiles") == 3
  db = dbutil.connect(abc_db_path)
  c = db.cursor()
  results = c.execute(
      'SELECT id,status,contents FROM PreprocessedFiles').fetchall()
  assert set([r[0] for r in results]) == {'a', 'b', 'c'}
  assert set([r[1] for r in results]) == {1}
  assert set([r[2] for r in results]) == {'bad code', }


def test_PreprocessDatabase_abc_internal_error(abc_db_path):
  """Test PreprocessDatabase with an internal error preprocessor."""
  preprocessors.PreprocessDatabase(abc_db_path, [
    'deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor',
    'deeplearning.clgen.preprocessors.preprocessors_test'
    ':MockPreprocessorInternalError'])
  assert dbutil.num_rows_in(abc_db_path, "PreprocessedFiles") == 3
  db = dbutil.connect(abc_db_path)
  c = db.cursor()
  results = c.execute(
      'SELECT id,status,contents FROM PreprocessedFiles').fetchall()
  assert set([r[0] for r in results]) == {'a', 'b', 'c'}
  assert set([r[1] for r in results]) == {1}
  assert set([r[2] for r in results]) == {'!!INTERNAL ERROR!! internal error', }


# Benchmarks.

def test_benchmark_GetPreprocessFunction_mock(benchmark):
  """Benchmark GetPreprocessFunction."""
  benchmark(preprocessors.GetPreprocessorFunction,
            'deeplearning.clgen.preprocessors.preprocessors_test'
            ':MockPreprocessor')


def test_benchmark_PreprocessDatabase_abc(benchmark, abc_db_path):
  """Benchmark PreprocessDatabase with a mock preprocessor."""
  benchmark(preprocessors.PreprocessDatabase, abc_db_path,

            ['deeplearning.clgen.preprocessors.preprocessors_test'
             ':MockPreprocessor'])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  logging.set_verbosity(logging.DEBUG)
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
