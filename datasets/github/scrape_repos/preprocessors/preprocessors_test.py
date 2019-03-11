"""Unit tests for //datasets/github/scrape_repos/preprocessors.py."""
import pathlib
import typing

import pytest

from datasets.github.scrape_repos.preprocessors import preprocessors
from datasets.github.scrape_repos.preprocessors import public
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def MakeFile(directory: pathlib.Path, relpath: str, contents: str) -> None:
  """Write contents to a file."""
  abspath = (directory / relpath).absolute()
  abspath.parent.mkdir(parents=True, exist_ok=True)
  with open(abspath, 'w') as f:
    f.write(contents)


@public.dataset_preprocessor
def MockPreprocessor(import_root: pathlib.Path, file_relpath: str, text: str,
                     all_file_relpaths: typing.List[str]) -> typing.List[str]:
  """A mock preprocessor."""
  del import_root
  del file_relpath
  del text
  del all_file_relpaths
  return ['PREPROCESSED']


@public.dataset_preprocessor
def MockPreprocessorError(
    import_root: pathlib.Path, file_relpath: str, text: str,
    all_file_relpaths: typing.List[str]) -> typing.List[str]:
  """A mock preprocessor which raises a ValueError."""
  del import_root
  del file_relpath
  del text
  del all_file_relpaths
  raise ValueError('ERROR')


def MockUndecoratedPreprocessor(
    import_root: pathlib.Path, file_relpath: str, text: str,
    all_file_relpaths: typing.List[str]) -> typing.List[str]:
  """A mock preprocessor which is not decorated with @dataset_preprocessor."""
  del import_root
  del file_relpath
  del text
  del all_file_relpaths
  return ['UNDECORATED']


# GetPreprocessFunction() tests.


def test_GetPreprocessFunction_empty_string():
  """Test that a ValueError is raised if no preprocessor is given."""
  with pytest.raises(ValueError) as e_info:
    preprocessors.GetPreprocessorFunction('')
  assert 'Invalid preprocessor name' in str(e_info.value)


def test_GetPreprocessFunction_missing_module():
  """Test that ValueError is raised if module not found."""
  with pytest.raises(ValueError) as e_info:
    preprocessors.GetPreprocessorFunction('not.a.real.module:Foo')
  assert 'not found' in str(e_info.value)


def test_GetPreprocessFunction_missing_function():
  """Test that ValueError is raised if module exists but function doesn't."""
  with pytest.raises(ValueError) as e_info:
    preprocessors.GetPreprocessorFunction(
        'datasets.github.scrape_repos.preprocessors.preprocessors_test:Foo')
  assert 'not found' in str(e_info.value)


def test_GetPreprocessFunction_undecorated_preprocessor():
  """Test that an ValueError is raised if preprocessor not decorated."""
  with pytest.raises(ValueError) as e_info:
    preprocessors.GetPreprocessorFunction(
        'datasets.github.scrape_repos.preprocessors.preprocessors_test'
        ':MockUndecoratedPreprocessor')
  assert '@dataset_preprocessor' in str(e_info.value)


def test_GetPreprocessFunction_mock_preprocessor():
  """Test that a mock preprocessor can be found."""
  f = preprocessors.GetPreprocessorFunction(
      'datasets.github.scrape_repos.preprocessors.preprocessors_test:MockPreprocessor'
  )
  assert f == MockPreprocessor


# Preprocess() tests.


def test_Preprocess_no_preprocessors(tempdir):
  """Test unmodified output if no preprocessors."""
  MakeFile(tempdir, 'a', 'hello')
  assert preprocessors.Preprocess(tempdir, 'a', ['a'], []) == ['hello']


def test_Preprocess_mock_preprocessor(tempdir):
  """Test unmodified output if no preprocessors."""
  MakeFile(tempdir, 'a', 'hello')
  assert preprocessors.Preprocess(tempdir, 'a', ['a'], [
      'datasets.github.scrape_repos.preprocessors.preprocessors_test'
      ':MockPreprocessor'
  ]) == ['PREPROCESSED']


def test_Preprocess_mock_preprocessor_exception(tempdir):
  """Test that an exception is propagated."""
  MakeFile(tempdir, 'a', 'hello')
  with pytest.raises(ValueError):
    preprocessors.Preprocess(tempdir, 'a', ['a'], [
        'datasets.github.scrape_repos.preprocessors.preprocessors_test'
        ':MockPreprocessorInternalError'
    ])


# Benchmarks.


def test_benchmark_GetPreprocessFunction_mock(benchmark):
  """Benchmark GetPreprocessFunction."""
  benchmark(
      preprocessors.GetPreprocessorFunction,
      'datasets.github.scrape_repos.preprocessors.preprocessors_test'
      ':MockPreprocessor')


if __name__ == '__main__':
  test.Main()
