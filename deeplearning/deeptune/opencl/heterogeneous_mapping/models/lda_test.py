"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping/models:lda."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from absl import flags

from deeplearning.deeptune.opencl.heterogeneous_mapping.models import lda
from experimental.compilers.reachability import llvm_util
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def g() -> nx.DiGraph:
  """Test fixture that returns a graph."""
  g = nx.DiGraph()
  g.add_node(0, inst2vec='foo')
  g.add_node(1, inst2vec='bar')
  g.add_node(2, inst2vec='car')
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  yield g


def test_Lda_ExtractGraphs_returns_cfgs(classify_df: pd.DataFrame):
  rows, graphs = zip(*lda.ExtractGraphs(classify_df[:3]))
  assert len(rows) == 3
  assert isinstance(graphs[0], llvm_util.LlvmControlFlowGraph)
  assert isinstance(graphs[1], llvm_util.LlvmControlFlowGraph)
  assert isinstance(graphs[2], llvm_util.LlvmControlFlowGraph)


def test_Lda_ExtractGraphs_cfgs_have_bytecode(classify_df: pd.DataFrame):
  rows, graphs = zip(*lda.ExtractGraphs(classify_df[:1]))
  assert len(rows) == 1
  graph = graphs[0]
  assert graph.graph['llvm_bytecode']


def test_Lda_GraphToInputTarget_input_graph_node_features(g: nx.DiGraph):
  """Test input graph node features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {'y_1hot': 'dar'}, g)

  assert input_graph.nodes[0]['features'] == 'foo'
  assert input_graph.nodes[1]['features'] == 'bar'
  assert input_graph.nodes[2]['features'] == 'car'


def test_Lda_GraphToInputTarget_target_graph_node_features(g: nx.DiGraph):
  """Test target graph node features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {'y_1hot': 'dar'}, g)

  np.testing.assert_array_almost_equal(
      target_graph.nodes[0]['features'], np.ones(1))
  np.testing.assert_array_almost_equal(
      target_graph.nodes[1]['features'], np.ones(1))
  np.testing.assert_array_almost_equal(
      target_graph.nodes[2]['features'], np.ones(1))


def test_Lda_GraphToInputTarget_input_graph_edge_features(g: nx.DiGraph):
  """Test input graph edge features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {'y_1hot': 'dar'}, g)

  np.testing.assert_array_almost_equal(
      input_graph.edges[0, 1]['features'], np.ones(1))
  np.testing.assert_array_almost_equal(
      input_graph.edges[1, 2]['features'], np.ones(1))


def test_Lda_GraphToInputTarget_target_graph_edge_features(g: nx.DiGraph):
  """Test target graph edge features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {'y_1hot': 'dar'}, g)

  np.testing.assert_array_almost_equal(
      target_graph.edges[0, 1]['features'], np.ones(1))
  np.testing.assert_array_almost_equal(
      target_graph.edges[1, 2]['features'], np.ones(1))


def test_Lda_GraphToInputTarget_input_graph_global_features(g: nx.DiGraph):
  """Test input graph global features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {'y_1hot': 'dar'}, g)

  np.testing.assert_array_almost_equal(
      input_graph.graph['features'], np.ones(1))


def test_Lda_GraphToInputTarget_target_graph_global_features(g: nx.DiGraph):
  """Test target graph global features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {'y_1hot': 'dar'}, g)

  assert target_graph.graph['features'] == 'dar'


if __name__ == '__main__':
  test.Main()
