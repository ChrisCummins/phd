"""Unit tests for //deeplearning/ml4pl/testing:random_graph_tuple_generator."""
from deeplearning.ml4pl.testing import random_graph_tuple_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@decorators.loop_for(seconds=2)
@test.Parametrize("node_count", (5, 10, 20))
def test_CreateRandomGraphTuple_node_count(node_count: int):
  """Test generating protos with specific node counts."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    node_count=node_count
  )
  assert graph_tuple.node_count == node_count


@decorators.loop_for(seconds=2)
@test.Parametrize("disjoint_graph_count", (1, 2, 3))
@test.Parametrize("node_x_dimensionality", (1, 2))
@test.Parametrize("node_y_dimensionality", (0, 1, 2))
@test.Parametrize("graph_x_dimensionality", (0, 1, 2))
@test.Parametrize("graph_y_dimensionality", (0, 1, 2))
def test_CreateRandomGraphTuple(
  disjoint_graph_count: int,
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
):
  """Black-box test of generator properties."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    disjoint_graph_count=disjoint_graph_count,
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )
  assert graph_tuple.disjoint_graph_count == disjoint_graph_count
  assert graph_tuple.node_x_dimensionality == node_x_dimensionality
  assert graph_tuple.node_y_dimensionality == node_y_dimensionality
  assert graph_tuple.graph_x_dimensionality == graph_x_dimensionality
  assert graph_tuple.graph_y_dimensionality == graph_y_dimensionality


def test_EnumerateTestSet():
  """Test the "real" protos."""
  protos = list(random_graph_tuple_generator.EnumerateTestSet())
  assert len(protos) == 100


def test_benchmark_CreateRandomGraphTuple(benchmark):
  """Benchmark graph tuple generation."""
  benchmark(random_graph_tuple_generator.CreateRandomGraphTuple)


if __name__ == "__main__":
  test.Main()
