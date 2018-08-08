"""Unit tests for //datasets/benchmarks/llvm_test_suite.py."""
import pathlib
import pytest
import sys
import typing
from absl import app
from absl import flags

from datasets.benchmarks import llvm_test_suite
from datasets.benchmarks.proto import benchmark_pb2


FLAGS = flags.FLAGS


@pytest.mark.parametrize('benchmark', llvm_test_suite.BENCHMARKS)
def test_benchmarks(benchmark: benchmark_pb2.Benchmark):
  """Test attributes of protos."""
  assert benchmark.name
  assert pathlib.Path(benchmark.binary).is_file()
  for path in benchmark.srcs:
    assert pathlib.Path(path).is_file()
  for path in benchmark.hdrs:
    assert pathlib.Path(path).is_file()


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
