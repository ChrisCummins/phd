"""Unit tests for //deeplearning/ml4pl/testing:random_networkx_generator."""
from deeplearning.ml4pl.testing import random_networkx_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@decorators.loop_for(seconds=2)
@test.Parametrize("node_x_dimensionality", (1, 2))
@test.Parametrize("node_y_dimensionality", (0, 1, 2))
@test.Parametrize("graph_x_dimensionality", (0, 1, 2))
@test.Parametrize("graph_y_dimensionality", (0, 1, 2))
def test_CreateRandomGraph(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
):
  """Black-box test of generator properties."""
  g = random_networkx_generator.CreateRandomGraph(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )
  assert len(g.nodes[0]["x"]) == node_x_dimensionality
  assert len(g.nodes[0]["y"]) == node_y_dimensionality
  assert len(g.graph["x"]) == graph_x_dimensionality
  assert len(g.graph["y"]) == graph_y_dimensionality


def test_EnumerateGraphTestSet():
  """Test the "real" protos."""
  protos = list(random_networkx_generator.EnumerateGraphTestSet())
  assert len(protos) == 100


def test_benchmark_CreateRandomGraph(benchmark):
  """Benchmark graph generation."""
  benchmark(random_networkx_generator.CreateRandomGraph)


if __name__ == "__main__":
  test.Main()
