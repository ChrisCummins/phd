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
"""Unit tests for //datasets/benchmarks/gpgpu:gpgpu.py."""
import pathlib
import typing

import pytest

from datasets.benchmarks.gpgpu import gpgpu
from datasets.benchmarks.gpgpu import gpgpu_pb2
from gpu.cldrive.legacy import env as cldrive_env
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

BENCHMARK_SUITES_TO_TEST = [
    gpgpu.DummyJustForTesting,
]


def test_RewriteClDeviceType_rewrites_file(tempdir: pathlib.Path):
  """Test that CL_DEVICE_TYPE is rewritten in file."""
  with open(tempdir / 'foo', 'w') as f:
    f.write("Hello world! The device type is: CL_DEVICE_TYPE_GPU.")
  gpgpu.RewriteClDeviceType(cldrive_env.OclgrindOpenCLEnvironment(), tempdir)
  with open(tempdir / 'foo') as f:
    assert f.read() == "Hello world! The device type is: CL_DEVICE_TYPE_CPU."


@pytest.mark.parametrize('benchmark_suite', BENCHMARK_SUITES_TO_TEST)
def test_BenchmarkSuite_path_contains_files(benchmark_suite: typing.Callable):
  """Test that benchmark suite contains files."""
  with benchmark_suite() as bs:
    assert bs.path.is_dir()
    assert list(bs.path.iterdir())


@pytest.mark.parametrize('benchmark_suite', BENCHMARK_SUITES_TO_TEST)
def test_BenchmarkSuite_invalid_path_access(benchmark_suite: typing.Callable):
  """Path cannot be accessed except when used as a context manager."""
  bs = benchmark_suite()
  with pytest.raises(TypeError):
    _ = bs.path


class MockBenchmarkObserver(gpgpu.BenchmarkRunObserver):

  def __init__(self, stop_after: int = 0):
    self.logs = []
    self.stop_after = stop_after

  def OnBenchmarkRun(self, log: gpgpu_pb2.GpgpuBenchmarkRun) -> bool:
    self.logs.append(log)
    return not (self.stop_after > 0 and len(self.logs) >= self.stop_after)


def test_MockBenchmarkObserver():
  observer = MockBenchmarkObserver(3)
  assert observer.OnBenchmarkRun('a')
  assert observer.OnBenchmarkRun('b')
  assert not observer.OnBenchmarkRun('c')
  assert observer.logs == ['a', 'b', 'c']


@pytest.mark.parametrize('benchmark_suite', BENCHMARK_SUITES_TO_TEST)
def test_BenchmarkSuite_integration_test(benchmark_suite: typing.Callable,
                                         tempdir: pathlib.Path):
  """Test compilation and execution of benchmark suite using oclgrind."""
  with benchmark_suite() as bs:
    bs.ForceOpenCLEnvironment(cldrive_env.OclgrindOpenCLEnvironment())
    observer = MockBenchmarkObserver(stop_after=1)

    # `stop_after` raises BenchmarkInterrupt.
    try:
      bs.Run([observer])
      assert False
    except gpgpu.BenchmarkInterrupt:
      pass

    assert len(observer.logs) == 1
    assert observer.logs[0].benchmark_name in bs.benchmarks


if __name__ == '__main__':
  test.Main()
