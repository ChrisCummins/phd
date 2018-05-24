"""Unit tests for //deeplearning/clgen/preprocessors/preprocessors.py."""
import sys
import typing

import pytest
from absl import app
from absl import logging

from deeplearning.clgen import errors
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


# Benchmarks.

def test_benchmark_GetPreprocessFunction_mock(benchmark):
  """Benchmark GetPreprocessFunction()"""
  benchmark(preprocessors.GetPreprocessorFunction,
            'deeplearning.clgen.preprocessors.preprocessors_test'
            ':MockPreprocessor')


def _PreprocessBenchmarkInnerLoop(preprocessors_: typing.List[str],
                                  code_in: str, code_out: str):
  """Benchmark inner loop."""
  assert preprocessors.Preprocess(code_in, preprocessors_) == code_out


def test_benchmark_Preprocess_opencl_pipeline(benchmark):
  """Benchmark Preprocess an OpenCL kernel using a full pipeline."""
  preprocessors_ = [
    'deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim',
    'deeplearning.clgen.preprocessors.opencl:Compile',
    'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
    'deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes',
    'deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines',
    'deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype',
    'deeplearning.clgen.preprocessors.common:StripTrailingWhitespace',
    'deeplearning.clgen.preprocessors.cxx:ClangFormat',
    'deeplearning.clgen.preprocessors.common:MinimumLineCount3', ]
  code_in = """
__kernel void foo(__global float* a, const int b) {
  int id = get_global_id(0);
  if (id <= b)
    a[id] = 0;
}
"""
  code_out = """\
kernel void A(global float* a, const int b) {
  int c = get_global_id(0);
  if (c <= b)
    a[c] = 0;
}\
"""
  benchmark(_PreprocessBenchmarkInnerLoop, preprocessors_, code_in, code_out)


def test_benchmark_Preprocess_cxx_pipeline(benchmark):
  """Benchmark Preprocess a C++ program using a full pipeline."""
  preprocessors_ = ['deeplearning.clgen.preprocessors.cxx:ClangPreprocess',
                    'deeplearning.clgen.preprocessors.cxx:Compile',
                    'deeplearning.clgen.preprocessors.cxx:NormalizeIdentifiers',
                    'deeplearning.clgen.preprocessors.common'
                    ':StripDuplicateEmptyLines',
                    'deeplearning.clgen.preprocessors.common:MinimumLineCount3',
                    'deeplearning.clgen.preprocessors.common'
                    ':StripTrailingWhitespace',
                    'deeplearning.clgen.preprocessors.cxx:ClangFormat', ]
  code_in = """
#include <iostream>

int do_something(int a) { return a * 2; }


int main(int argc, char **argv) { return do_something(argc); }
"""
  code_out = """\
int A(int a) {
  return a * 2;
}

int B(int a, char** b) {
  return A(a);
}\
"""
  benchmark(_PreprocessBenchmarkInnerLoop, preprocessors_, code_in, code_out)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  logging.set_verbosity(logging.DEBUG)
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
