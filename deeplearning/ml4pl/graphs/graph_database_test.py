"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
import pathlib
import pickle

import networkx as nx
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs import graph_database

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> graph_database.Database:
  """Fixture that returns an sqlite database."""
  yield graph_database.Database(f'sqlite:///{tempdir}/db')


def test_Graph_pickled_networkx_graph(db: graph_database.Database):
  """Test saving and loading graph with a pickled networkx graph of data."""
  g = nx.MultiDiGraph()
  g.add_edge('A', 'B')
  g.add_edge('B', 'C')

  with db.Session(commit=True) as s:
    s.add(
        graph_database.GraphMeta(
            group="train",
            bytecode_id=1,
            source_name='foo',
            relpath='bar',
            language='c',
            node_count=g.number_of_nodes(),
            edge_count=g.number_of_edges(),
            graph=graph_database.Graph(data=pickle.dumps(g))))
  with db.Session() as s:
    gm = s.query(graph_database.GraphMeta).first()
    assert gm.group == "train"
    assert gm.bytecode_id == 1
    assert gm.source_name == 'foo'
    assert gm.relpath == 'bar'
    assert gm.language == 'c'
    assert gm.node_count == 3
    assert gm.edge_count == 2

    g = pickle.loads(gm.graph.data)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2

    assert gm.id == gm.graph.id


def test_Graph_pickled_dictionary(db: graph_database.Database):
  """Test saving and loading graph with a pickled dictionary of data."""
  with db.Session(commit=True) as s:
    s.add(
        graph_database.GraphMeta(
            group="train",
            bytecode_id=1,
            source_name='foo',
            relpath='bar',
            language='c',
            node_count=1,
            edge_count=2,
            graph=graph_database.Graph(data=pickle.dumps({
                "a": 1,
                "b": 2
            }))))
  with db.Session() as s:
    gm = s.query(graph_database.GraphMeta).first()
    assert gm.bytecode_id == 1

    d = pickle.loads(gm.graph.data)
    assert d["a"] == 1
    assert d["b"] == 2

    assert gm.id == gm.graph.id


@pytest.fixture(scope='function')
def graph() -> nx.MultiDiGraph:
  g = nx.MultiDiGraph()
  g.bytecode_id = 1
  g.source_name = 'foo'
  g.relpath = 'bar'
  g.language = 'c'
  g.x = [0, 0, 0, 0]
  g.data_flow_max_steps_required = 10
  g.add_node('A', type='statement', x=[0, 1], y=[1])
  g.add_node('B', type='statement', x=[0, 1], y=[1])
  g.add_node('C', type='statement', x=[0, 1], y=[1])
  g.add_node('D', type='statement', x=[0, 1], y=[1])
  g.add_node('root', type='magic', x=[0, 1], y=[1])
  g.add_edge('A', 'B', flow='control', fx=[0, 0, 0], fy=[2, 2])
  g.add_edge('B', 'C', flow='control', fx=[0, 0, 0], fy=[2, 2])
  g.add_edge('C', 'D', flow='control', fx=[0, 0, 0], fy=[2, 2])
  g.add_edge('root', 'A', flow='call', fx=[0, 0, 0], fy=[2, 2])
  g.add_edge('A', 'D', flow='data', fx=[0, 0, 0], fy=[2, 2])
  return g


def test_Graph_CreateWithGraphDict(graph: nx.MultiDiGraph):
  """Test column values created by CreateWithGraphDict()."""
  gm = graph_database.GraphMeta.CreateWithGraphDict(graph,
                                                    {'control', 'call', 'data'},
                                                    edge_x='fx',
                                                    edge_y='fy')

  assert gm.bytecode_id == 1
  assert gm.source_name == 'foo'
  assert gm.relpath == 'bar'
  assert gm.language == 'c'
  assert gm.node_count == 5
  assert gm.edge_count == 5
  assert gm.edge_type_count == 3
  assert gm.node_features_dimensionality == 2
  assert gm.node_labels_dimensionality == 1
  assert gm.edge_features_dimensionality == 3
  assert gm.edge_labels_dimensionality == 2
  assert gm.graph_features_dimensionality == 4
  assert gm.graph_labels_dimensionality == 0
  assert gm.data_flow_max_steps_required == 10
  assert gm.graph.data


def test_benchmark_Graph_CreateWithGraphDict(benchmark, graph: nx.MultiDiGraph):
  """Benchmark CreateWithGraphDict()."""
  benchmark(graph_database.GraphMeta.CreateWithGraphDict,
            graph, {'control', 'call', 'data'},
            edge_x='fx',
            edge_y='fy')


if __name__ == '__main__':
  test.Main()
