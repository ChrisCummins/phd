# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //deeplearning/ml4pl/graphs/py:random_graph_builder."""
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.py import random_graph_builder
from labm8.py import test

FLAGS = test.FLAGS


def test_benchmark(benchmark):
  """Micro-benchmark graph generation."""

  def MakeOne():
    builder = random_graph_builder.RandomGraphBuilder()
    return builder.GetSerializedGraphProto()

  benchmark(MakeOne)


def test_benchmark_deserialize(benchmark):
  """Micro-benchmark graph generation."""
  proto = programl_pb2.ProgramGraphProto()

  def MakeOne():
    builder = random_graph_builder.RandomGraphBuilder()
    proto.ParseFromString(builder.GetSerializedGraphProto())

  benchmark(MakeOne)


if __name__ == "__main__":
  test.Main()
