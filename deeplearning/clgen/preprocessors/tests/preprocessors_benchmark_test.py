# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Benchmarks for the preprocessing pipeline."""
import typing

import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import preprocessors
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = "deeplearning.clgen"

# A full preprocessing pipeline for the C++ programming language.
CXX_PREPROCESSORS = [
  "deeplearning.clgen.preprocessors.cxx:ClangPreprocess",
  "deeplearning.clgen.preprocessors.cxx:Compile",
  "deeplearning.clgen.preprocessors.cxx" ":NormalizeIdentifiers",
  "deeplearning.clgen.preprocessors.common" ":StripDuplicateEmptyLines",
  "deeplearning.clgen.preprocessors.common" ":MinimumLineCount3",
  "deeplearning.clgen.preprocessors.common" ":StripTrailingWhitespace",
  "deeplearning.clgen.preprocessors.cxx:ClangFormat",
]
# A full preprocessing pipeline for the OpenCL programming language.
OPENCL_PREPROCESSORS = [
  "deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim",
  "deeplearning.clgen.preprocessors.opencl:Compile",
  "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers",
  "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes",
  "deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines",
  "deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype",
  "deeplearning.clgen.preprocessors.common:StripTrailingWhitespace",
  "deeplearning.clgen.preprocessors.opencl:ClangFormat",
  "deeplearning.clgen.preprocessors.common:MinimumLineCount3",
]


def _PreprocessBenchmarkInnerLoop(
  preprocessors_: typing.List[str], code_in: str, code_out: str
):
  """Benchmark inner loop for code with expected output."""
  assert preprocessors.Preprocess(code_in, preprocessors_) == code_out


def _PreprocessBenchmarkInnerLoopBadCode(
  preprocessors_: typing.List[str], code_in
):
  """Benchmark inner loop for bad code."""
  with test.Raises(errors.BadCodeException):
    preprocessors.Preprocess(code_in, preprocessors_)


def test_benchmark_cxx_small_program(benchmark):
  """Benchmark preprocessing a C++ program using a full pipeline."""
  code_in = """
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
  benchmark(_PreprocessBenchmarkInnerLoop, CXX_PREPROCESSORS, code_in, code_out)


def test_benchmark_cxx_invalid_syntax(benchmark):
  """Benchmark preprocessing a C++ program with syntax errors."""
  benchmark(
    _PreprocessBenchmarkInnerLoopBadCode, CXX_PREPROCESSORS, "inva@asd!!!"
  )


def test_benchmark_opencl_small_program(benchmark):
  """Benchmark preprocessing an OpenCL kernel using a full pipeline."""
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
  benchmark(
    _PreprocessBenchmarkInnerLoop, OPENCL_PREPROCESSORS, code_in, code_out
  )


def test_benchmark_opencl_invalid_syntax(benchmark):
  """Benchmark preprocessing an OpenCL program with syntax errors."""
  benchmark(
    _PreprocessBenchmarkInnerLoopBadCode, OPENCL_PREPROCESSORS, "inva@asd!!!"
  )


if __name__ == "__main__":
  test.Main()
