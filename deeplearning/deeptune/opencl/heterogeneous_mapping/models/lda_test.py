"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping/models:lda."""
import numpy as np
import pandas as pd
import pytest
from absl import flags

from deeplearning.deeptune.opencl.heterogeneous_mapping.models import lda
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import testlib
from experimental.compilers.reachability import llvm_util
from labm8 import test

FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def g() -> llvm_util.LlvmControlFlowGraph:
  """Test fixture that returns a graph."""
  g = llvm_util.LlvmControlFlowGraph()
  g.add_node(0, inst2vec=np.array([1, 2, 3]), entry=True)
  g.add_node(1, inst2vec=np.array([4, 5, 6]))
  g.add_node(2, inst2vec=np.array([7, 8, 9]), exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  yield g


def test_Lda_ExtractGraphs_returns_cfgs(classify_df: pd.DataFrame):
  """Test that CFGs are returned."""
  rows, graphs = zip(*lda.Lda.ExtractGraphs(classify_df[:3]))
  assert len(rows) == 3
  assert isinstance(graphs[0], llvm_util.LlvmControlFlowGraph)
  assert isinstance(graphs[1], llvm_util.LlvmControlFlowGraph)
  assert isinstance(graphs[2], llvm_util.LlvmControlFlowGraph)


def test_Lda_ExtractGraphs_cfgs_have_bytecode(single_program_df: pd.DataFrame):
  """Test that CFG has bytecode set."""
  rows, graphs = zip(
      *lda.Lda.ExtractGraphs(lda.SetNormalizedColumns(single_program_df)))
  assert len(rows) == 1
  assert graphs[0].graph['llvm_bytecode']


def test_Lda_EncodeGraphs_inst2vec_vectors(single_program_df: pd.DataFrame):
  """Test that CFG has inst2vec attribute set."""
  model = lda.Lda()
  rows, graphs = zip(*model.EncodeGraphs(
      model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df))))
  assert len(rows) == 1
  assert len(graphs[0].nodes)

  for _, data in graphs[0].nodes(data=True):
    assert 'inst2vec' in data
    # Check the shape of the embedding vector.
    assert data['inst2vec'].shape == (model.embedding_dim,)


def test_Lda_EncodeGraphs_inst2vec_encoded(single_program_df: pd.DataFrame):
  """Test that CFG has inst2vec_encoded attribute set."""
  model = lda.Lda()
  rows, graphs = zip(*model.EncodeGraphs(
      model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df))))
  assert len(rows) == 1
  assert len(graphs[0].nodes)

  for _, data in graphs[0].nodes(data=True):
    assert 'inst2vec_encoded' in data
    # Vocabulary elements are non-negative integers.
    assert data['inst2vec_encoded'] > 0


def test_Lda_EncodeGraphs_num_unknown_statements(
    single_program_df: pd.DataFrame):
  """Test that num_unknown_statements is set on graph."""
  model = lda.Lda()
  rows, graphs = zip(*model.EncodeGraphs(
      model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df))))
  assert len(rows) == 1

  assert graphs[0].graph['num_unknown_statements'] >= 0


def test_Lda_EncodeGraphs_unique_encoded(single_program_df: pd.DataFrame):
  """Test that CFG has multiple unique encoded nodes."""
  model = lda.Lda()
  rows, graphs = zip(*model.EncodeGraphs(
      model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df))))
  assert len(rows) == 1
  assert len(graphs[0].nodes)

  uniq_encoded = set(
      data['inst2vec_encoded'] for _, data in graphs[0].nodes(data=True))

  assert len(uniq_encoded) > 1


def test_Lda_GraphToInputTarget_input_graph_node_features(
    g: llvm_util.LlvmControlFlowGraph):
  """Test input graph node features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {
          'y_1hot': np.array([0, 1]),
          'target_gpu_name': 'a',
          'feature:a:transfer_norm': .1,
          'param:a:wgsize_norm': 0.5
      }, g)

  # Each feature vector is a concatenation of [entry, exit, inst2vec]
  np.testing.assert_array_almost_equal(input_graph.nodes[0]['features'],
                                       np.array([1, 0, 1, 2, 3]))
  np.testing.assert_array_almost_equal(input_graph.nodes[1]['features'],
                                       np.array([0, 0, 4, 5, 6]))
  np.testing.assert_array_almost_equal(input_graph.nodes[2]['features'],
                                       np.array([0, 1, 7, 8, 9]))


def test_Lda_GraphToInputTarget_target_graph_node_features(
    g: llvm_util.LlvmControlFlowGraph):
  """Test target graph node features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {
          'y_1hot': np.array([0, 1]),
          'target_gpu_name': 'a',
          'feature:a:transfer_norm': .1,
          'param:a:wgsize_norm': 0.5
      }, g)

  np.testing.assert_array_almost_equal(target_graph.nodes[0]['features'],
                                       np.ones(1))
  np.testing.assert_array_almost_equal(target_graph.nodes[1]['features'],
                                       np.ones(1))
  np.testing.assert_array_almost_equal(target_graph.nodes[2]['features'],
                                       np.ones(1))


def test_Lda_GraphToInputTarget_input_graph_edge_features(
    g: llvm_util.LlvmControlFlowGraph):
  """Test input graph edge features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {
          'y_1hot': np.array([0, 1]),
          'target_gpu_name': 'a',
          'feature:a:transfer_norm': .1,
          'param:a:wgsize_norm': 0.5
      }, g)

  np.testing.assert_array_almost_equal(input_graph.edges[0, 1]['features'],
                                       np.ones(1))
  np.testing.assert_array_almost_equal(input_graph.edges[1, 2]['features'],
                                       np.ones(1))


def test_Lda_GraphToInputTarget_target_graph_edge_features(
    g: llvm_util.LlvmControlFlowGraph):
  """Test target graph edge features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {
          'y_1hot': np.array([0, 1]),
          'target_gpu_name': 'a',
          'feature:a:transfer_norm': .1,
          'param:a:wgsize_norm': 0.5
      }, g)

  np.testing.assert_array_almost_equal(target_graph.edges[0, 1]['features'],
                                       np.ones(1))
  np.testing.assert_array_almost_equal(target_graph.edges[1, 2]['features'],
                                       np.ones(1))


def test_Lda_GraphToInputTarget_input_graph_global_features(
    g: llvm_util.LlvmControlFlowGraph):
  """Test input graph global features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {
          'y_1hot': np.array([0, 1]),
          'target_gpu_name': 'a',
          'feature:a:transfer_norm': .1,
          'param:a:wgsize_norm': 0.5
      }, g)

  np.testing.assert_array_almost_equal(input_graph.graph['features'], [.1, .5])


def test_Lda_GraphToInputTarget_target_graph_global_features(
    g: llvm_util.LlvmControlFlowGraph):
  """Test target graph global features."""
  input_graph, target_graph = lda.Lda.GraphToInputTarget(
      {
          'y_1hot': np.array([0, 1]),
          'target_gpu_name': 'a',
          'feature:a:transfer_norm': .1,
          'param:a:wgsize_norm': 0.5
      }, g)

  np.testing.assert_array_almost_equal(target_graph.graph['features'],
                                       np.array([0, 1]))


def test_Lda_GraphsToInputTargets_node_features_shape(
    single_program_df: pd.DataFrame):
  """Test that node features have correct shape."""
  model = lda.Lda()
  input_graphs, target_graphs = zip(*model.GraphsToInputTargets(
      model.EncodeGraphs(
          model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df)))))
  assert len(input_graphs) == 1
  assert input_graphs[0].nodes[0]['features'].shape == (
      model.embedding_dim + 2,)
  assert target_graphs[0].nodes[0]['features'].shape == (1,)


def test_Lda_GraphsToInputTargets_node_features_entry_exit_blocks(
    single_program_df: pd.DataFrame):
  """Assert that only a single entry node is set."""
  model = lda.Lda()
  input_graphs, target_graphs = zip(*model.GraphsToInputTargets(
      model.EncodeGraphs(
          model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df)))))
  assert len(input_graphs) == 1
  node_features = np.vstack(
      [d['features'] for _, d in input_graphs[0].nodes(data=True)])
  # The 'entry' and 'exit' blocks are the first two elements of the feature
  # vector. Only a single value in each column should be set.
  assert sum(node_features[:, 0]) == pytest.approx(1)
  assert sum(node_features[:, 1]) == pytest.approx(1)


def test_Lda_GraphsToInputTargets_node_features_dtype(
    single_program_df: pd.DataFrame):
  """Test that node features have correct type."""
  model = lda.Lda()
  input_graphs, target_graphs = zip(*model.GraphsToInputTargets(
      model.EncodeGraphs(
          model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df)))))
  assert len(input_graphs) == 1
  assert input_graphs[0].nodes[0]['features'].dtype == np.float32
  assert target_graphs[0].nodes[0]['features'].dtype == np.float32


def test_Lda_GraphsToInputTargets_global_features_shape(
    single_program_df: pd.DataFrame):
  """Test that node features have correct shape."""
  model = lda.Lda()
  input_graphs, target_graphs = zip(*model.GraphsToInputTargets(
      model.EncodeGraphs(
          model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df)))))
  assert len(input_graphs) == 1
  assert input_graphs[0].graph['features'].shape == (2,)
  assert target_graphs[0].graph['features'].shape == (2,)


def test_Lda_GraphsToInputTargets_global_features_dtype(
    single_program_df: pd.DataFrame):
  """Test that graph features have correct type."""
  model = lda.Lda()
  input_graphs, target_graphs = zip(*model.GraphsToInputTargets(
      model.EncodeGraphs(
          model.ExtractGraphs(lda.SetNormalizedColumns(single_program_df)))))
  assert len(input_graphs) == 1
  assert input_graphs[0].graph['features'].dtype == np.float32
  assert target_graphs[0].graph['features'].dtype == np.float32


def test_model(classify_df, classify_df_atomizer):
  """Run common model tests."""
  testlib.HeterogeneousMappingModelTest(lda.Lda, classify_df,
                                        classify_df_atomizer, {})


if __name__ == '__main__':
  test.Main()
