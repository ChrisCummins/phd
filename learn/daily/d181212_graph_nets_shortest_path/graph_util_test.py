"""Unit tests for //learn/daily/d181212_graph_nets_shortest_path:graph_util."""
import sys
import typing

import networkx as nx
import numpy as np
import pytest
from absl import app
from absl import flags

from learn.daily.d181212_graph_nets_shortest_path import graph_util


FLAGS = flags.FLAGS


def test_GenerateGraph_return_type():
  """Test that networkx Graph is returned."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert isinstance(graph, nx.Graph)


def test_GenerateGraph_num_nodes():
  """Test that number of nodes is within requested range."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert 10 <= graph.number_of_nodes() < 15


def test_GenerateGraph_num_edges():
  """Test that graph has at least one edge per node."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert graph.number_of_edges() >= graph.number_of_nodes()


def test_GenerateGraph_is_connected():
  """Test that graph is connected."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert nx.is_connected(graph)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
