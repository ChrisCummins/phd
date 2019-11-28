# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //datasets/benchmarks/llvm_test_suite.py."""
import pathlib
import tempfile

import pytest

from compilers.llvm import clang
from datasets.benchmarks import llvm_test_suite
from datasets.benchmarks.proto import benchmarks_pb2
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = "datasets.benchmarks"


@test.Parametrize("benchmark", llvm_test_suite.BENCHMARKS)
def test_benchmarks(benchmark: benchmarks_pb2.Benchmark):
  """Test attributes of protos."""
  assert benchmark.name
  assert pathlib.Path(benchmark.binary).is_file()
  for path in benchmark.srcs:
    assert pathlib.Path(path).is_file()
  for path in benchmark.hdrs:
    assert pathlib.Path(path).is_file()
  # Compile the sources.
  with tempfile.TemporaryDirectory() as d:
    clang.Compile(
      [pathlib.Path(x) for x in benchmark.srcs], pathlib.Path(d) / "exe"
    )
    assert (pathlib.Path(d) / "exe").is_file()


if __name__ == "__main__":
  test.Main()
