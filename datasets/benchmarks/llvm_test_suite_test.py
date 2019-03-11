"""Unit tests for //datasets/benchmarks/llvm_test_suite.py."""
import pathlib
import tempfile

import pytest

from compilers.llvm import clang
from datasets.benchmarks import llvm_test_suite
from datasets.benchmarks.proto import benchmarks_pb2
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


@pytest.mark.parametrize('benchmark', llvm_test_suite.BENCHMARKS)
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
    clang.Compile([pathlib.Path(x) for x in benchmark.srcs],
                  pathlib.Path(d) / 'exe')
    assert (pathlib.Path(d) / 'exe').is_file()


if __name__ == '__main__':
  test.Main()
