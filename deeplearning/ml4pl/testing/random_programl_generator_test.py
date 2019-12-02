"""Unit tests for //deeplearning/ml4pl/testing:random_programl_generator."""
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


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


def test_EnumerateProtoTestSet():
  """Test the "real" protos."""
  protos = list(random_programl_generator.EnumerateProtoTestSet())
  assert len(protos) == 100


def test_benchmark_CreateRandomProto(benchmark):
  """Benchmark proto generation."""
  benchmark(random_programl_generator.CreateRandomProto)


if __name__ == "__main__":
  test.Main()
