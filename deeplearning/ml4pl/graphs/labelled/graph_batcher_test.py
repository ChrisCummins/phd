"""Unit tests for //deeplearning/ml4pl/graphs/labelled/graph_batcher."""
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import graph_batcher
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def _MakeIterator(graphs):
  for graph in graphs:
    yield graph


def test_NextGraph_empty_list():
  """Empty iterator returns none."""
  options = graph_batcher.GraphBatchOptions(max_nodes=100)
  graph = graph_batcher.GraphBatch.NextGraph(_MakeIterator([]), options)
  assert graph is None


def test_NextGraph_larger_than_batch_size():
  """Error is raised when graph is larger than batch size."""
  big_graph = graph_database.GraphMeta(node_count=10000)

  options = graph_batcher.GraphBatchOptions(max_nodes=100)
  with test.Raises(ValueError):
    graph_batcher.GraphBatch.NextGraph(_MakeIterator([big_graph]), options)

  options = graph_batcher.GraphBatchOptions(max_nodes=9999999)
  graph = graph_batcher.GraphBatch.NextGraph(
    _MakeIterator([big_graph]), options
  )
  assert graph.node_count == big_graph.node_count


def test_GraphBatchOptions_ShouldAddToBatch_no_filters():
  options = graph_batcher.GraphBatchOptions(max_graphs=0, max_nodes=0)
  log = log_database.BatchLogMeta(node_count=0, graph_count=0)
  graph = graph_database.GraphMeta(node_count=0, graph=graph_database.Graph())
  assert options.ShouldAddToBatch(graph, log)


def test_GraphBatchOptions_ShouldAddToBatch_no_data():
  options = graph_batcher.GraphBatchOptions(max_graphs=0, max_nodes=0)
  log = log_database.BatchLogMeta(node_count=0, graph_count=0)
  graph = graph_database.GraphMeta(node_count=0, graph=None)
  assert not options.ShouldAddToBatch(graph, log)


def test_GraphBatchOptions_ShouldAddToBatch_node_count_filter():
  options = graph_batcher.GraphBatchOptions(max_graphs=0, max_nodes=100)
  log = log_database.BatchLogMeta(node_count=30, graph_count=0)
  graph = graph_database.GraphMeta(node_count=50, graph=graph_database.Graph())
  assert options.ShouldAddToBatch(graph, log)

  graph = graph_database.GraphMeta(node_count=80)
  assert not options.ShouldAddToBatch(graph, log)


def test_GraphBatchOptions_ShouldAddToBatch_graph_count_filter():
  options = graph_batcher.GraphBatchOptions(max_graphs=10, max_nodes=0)
  log = log_database.BatchLogMeta(node_count=0, graph_count=0)
  graph = graph_database.GraphMeta(node_count=0, graph=graph_database.Graph())
  assert options.ShouldAddToBatch(graph, log)

  log.graph_count = 10
  assert not options.ShouldAddToBatch(graph, log)


if __name__ == "__main__":
  test.Main()
