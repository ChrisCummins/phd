"""Unit tests for //deeplearning/ml4pl/graphs/labelled/graph_batcher."""
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher

FLAGS = app.FLAGS


def _MakeIterator(graphs):
  for graph in graphs:
    yield graph


def test_NextGraph_empty_list():
  """Empty iterator returns none."""
  graph = graph_batcher.GraphBatch.NextGraph(_MakeIterator([]))
  assert graph is None


def test_NextGraph_larger_than_batch_size():
  """Error is raised when graph is larger than batch size."""
  big_graph = graph_database.GraphMeta(node_count=10000)

  FLAGS.batch_size = 100
  with pytest.raises(ValueError):
    graph_batcher.GraphBatch.NextGraph(_MakeIterator([big_graph]))

  FLAGS.batch_size = 999999
  graph = graph_batcher.GraphBatch.NextGraph(_MakeIterator([big_graph]))
  assert graph.node_count == big_graph.node_count


if __name__ == '__main__':
  test.Main()
