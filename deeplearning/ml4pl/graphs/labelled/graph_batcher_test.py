"""Unit tests for //deeplearning/ml4pl/graphs/labelled:graph_batcher."""
from typing import Iterable
from typing import List

from deeplearning.ml4pl.graphs.labelled import graph_batcher
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.testing import random_graph_tuple_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


def MockIterator(
  graphs: List[graph_tuple.GraphTuple],
) -> Iterable[graph_tuple.GraphTuple]:
  """Return an iterator over graphs."""
  for graph in graphs:
    yield graph


def test_GraphBatcher_empty_graphs_list():
  """Test input with empty graph """
  batcher = graph_batcher.GraphBatcher(MockIterator([]))
  with test.Raises(StopIteration):
    next(batcher)


@test.Parametrize("graph_count", (1, 5, 10))
def test_GraphBatcher_collect_all_inputs(graph_count: int):
  batcher = graph_batcher.GraphBatcher(
    MockIterator(
      [
        random_graph_tuple_generator.CreateRandomGraphTuple()
        for _ in range(graph_count)
      ]
    )
  )
  batches = list(batcher)
  assert len(batches) == 1
  assert batches[0].is_disjoint_graph
  assert batches[0].disjoint_graph_count == graph_count


def test_GraphBatcher_max_node_count_limit_handler_error():
  """Test that error is raised when graph is larger than max node count."""
  big_graph = random_graph_tuple_generator.CreateRandomGraphTuple(node_count=10)

  batcher = graph_batcher.GraphBatcher(
    MockIterator([big_graph]),
    max_node_count=5,
    max_node_count_limit_handler="error",
  )

  with test.Raises(ValueError):
    next(batcher)


def test_GraphBatcher_max_node_count_limit_handler_skip():
  """Test that graph is included when larger than max node count."""
  big_graph = random_graph_tuple_generator.CreateRandomGraphTuple(node_count=10)

  batcher = graph_batcher.GraphBatcher(
    MockIterator([big_graph]),
    max_node_count=5,
    max_node_count_limit_handler="include",
  )

  assert next(batcher)


def test_GraphBatcher_max_node_count_limit_handler_skip():
  """Test that graph is skipped when larger than max node count."""
  big_graph = random_graph_tuple_generator.CreateRandomGraphTuple(node_count=10)

  batcher = graph_batcher.GraphBatcher(
    MockIterator([big_graph]),
    max_node_count=5,
    max_node_count_limit_handler="skip",
  )

  try:
    next(batcher)
  except StopIteration:
    pass


def test_GraphBatcher_divisible_node_count():
  """Test the number of batches returned with evenly divisible node counts."""
  batcher = graph_batcher.GraphBatcher(
    MockIterator(
      [
        random_graph_tuple_generator.CreateRandomGraphTuple(node_count=5),
        random_graph_tuple_generator.CreateRandomGraphTuple(node_count=5),
        random_graph_tuple_generator.CreateRandomGraphTuple(node_count=5),
        random_graph_tuple_generator.CreateRandomGraphTuple(node_count=5),
      ]
    ),
    max_node_count=10,
  )

  batches = list(batcher)
  assert len(batches) == 2
  assert batches[0].is_disjoint_graph
  assert batches[0].disjoint_graph_count == 2
  assert batches[1].is_disjoint_graph
  assert batches[1].disjoint_graph_count == 2


def test_GraphBatcher_max_graph_count():
  """Test the number of batches when max graphs are filtered."""
  batcher = graph_batcher.GraphBatcher(
    MockIterator(
      [random_graph_tuple_generator.CreateRandomGraphTuple() for _ in range(7)]
    ),
    max_graph_count=3,
  )

  batches = list(batcher)
  assert len(batches) == 3
  assert batches[0].disjoint_graph_count == 3
  assert batches[1].disjoint_graph_count == 3
  assert batches[2].disjoint_graph_count == 1


def test_GraphBatcher_exact_graph_count():
  """Test the number of batches when exact graph counts are required."""
  batcher = graph_batcher.GraphBatcher(
    MockIterator(
      [random_graph_tuple_generator.CreateRandomGraphTuple() for _ in range(7)]
    ),
    exact_graph_count=3,
  )

  batches = list(batcher)
  assert len(batches) == 2
  assert batches[0].disjoint_graph_count == 3
  assert batches[1].disjoint_graph_count == 3
  # The last graph is ignored because we have exactly the right number of
  # graphs.


@decorators.loop_for(seconds=5)
@test.Parametrize("graph_count", (1, 10, 100))
@test.Parametrize("max_node_count", (50, 100))
@test.Parametrize("max_graph_count", (0, 3, 10))
def test_fuzz_GraphBatcher(
  graph_count: int, max_graph_count: int, max_node_count: int
):
  """Fuzz the graph batcher with a range of parameter choices and input
  sizes.
  """
  graphs = MockIterator(
    [
      random_graph_tuple_generator.CreateRandomGraphTuple()
      for _ in range(graph_count)
    ]
  )
  batcher = graph_batcher.GraphBatcher(
    graphs, max_node_count=max_node_count, max_graph_count=max_graph_count
  )
  batches = list(batcher)
  assert sum(b.disjoint_graph_count for b in batches) == graph_count


if __name__ == "__main__":
  test.Main()
