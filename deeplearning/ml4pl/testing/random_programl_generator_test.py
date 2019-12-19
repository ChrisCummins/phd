# Copyright 2019 the ProGraML authors.
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
"""Unit tests for //deeplearning/ml4pl/testing:random_programl_generator."""
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@decorators.loop_for(seconds=2)
@test.Parametrize("node_count", (5, 10, 20))
def test_CreateRandomProto_node_count(node_count: int):
  """Test generating protos with specific node counts."""
  proto = random_programl_generator.CreateRandomProto(node_count=node_count)
  assert len(proto.node) == node_count


@decorators.loop_for(seconds=2)
@test.Parametrize("node_x_dimensionality", (1, 2))
@test.Parametrize("node_y_dimensionality", (0, 1, 2))
@test.Parametrize("graph_x_dimensionality", (0, 1, 2))
@test.Parametrize("graph_y_dimensionality", (0, 1, 2))
def test_CreateRandomProto(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
):
  """Black-box test of generator properties."""
  proto = random_programl_generator.CreateRandomProto(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )
  assert proto.IsInitialized()
  for node in proto.node:
    assert len(node.x) == node_x_dimensionality
    assert len(node.y) == node_y_dimensionality
    if node.function:
      assert node.function <= len(proto.function)
  for edge in proto.edge:
    assert edge.source_node < len(proto.node)
    assert edge.destination_node < len(proto.node)
  assert len(proto.x) == graph_x_dimensionality
  assert len(proto.y) == graph_y_dimensionality


def test_EnumerateTestSet():
  """Test the "real" protos."""
  protos = list(random_programl_generator.EnumerateTestSet())
  assert len(protos) == 100


def test_benchmark_CreateRandomProto(benchmark):
  """Benchmark proto generation."""
  benchmark(random_programl_generator.CreateRandomProto)


if __name__ == "__main__":
  test.Main()
