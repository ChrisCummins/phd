"""Unit tests for //experimental/compilers/reachability/ggnn:graph_database."""
import networkx as nx
import pathlib
import pickle
import pytest

from experimental.compilers.reachability.ggnn import graph_database
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS

@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> graph_database.Database:
  yield graph_database.Database(f'sqlite:///{tempdir}/db')


def test_Graph_pickled_networkx_graph(db: graph_database.Database):
  """Test saving and loading graph with a pickled networkx graph of data."""
  g = nx.MultiDiGraph()
  g.add_edge('A', 'B')
  g.add_edge('B', 'C')

  with db.Session(commit=True) as s:
    s.add(graph_database.GraphMeta(
        group="train",
        bytecode_id = 1,
        source_name = 'foo',
        relpath = 'bar',
        language = 'c',
        node_count = g.number_of_nodes(),
        edge_count = g.number_of_edges(),
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
    s.add(graph_database.GraphMeta(
        group = "train",
        bytecode_id = 1,
        source_name = 'foo',
        relpath = 'bar',
        language = 'c',
        node_count = 1,
        edge_count = 2,
        graph=graph_database.Graph(data=pickle.dumps({"a": 1, "b": 2}))))
  with db.Session() as s:
    gm = s.query(graph_database.GraphMeta).first()
    assert gm.bytecode_id == 1

    d = pickle.loads(gm.graph.data)
    assert d["a"] == 1
    assert d["b"] == 2

    assert gm.id == gm.graph.id


def test_BufferedGraphDatabaseReader(db: graph_database.Database):
  def _MakeGraphMeta(i):
    return graph_database.GraphMeta(
        group = "train",
        bytecode_id = 1,
        source_name = 'foo',
        relpath = 'bar',
        language = 'c',
        node_count = i,
        edge_count = 2,
        graph=graph_database.Graph(data=pickle.dumps({"a": 1})))

  with db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  graphs = []
  for graph in graph_database.BufferedGraphReader(db, buffer_size=10):
    graphs.append(graph)
  assert len(graphs) == 512
  assert all([g.bytecode_id == 1 for g in graphs])

  # Test with a filter to retrieve only those with even numbers of nodes.
  graphs = []
  filter_cb = lambda: graph_database.GraphMeta.node_count % 2 == 0
  for graph in graph_database.BufferedGraphReader(
      db, filters=[filter_cb], buffer_size=10):
    graphs.append(graph)
  assert len(graphs) == 256

  # Test with a random ordering.
  graphs = list(graph_database.BufferedGraphReader(db, order_by_random=True))
  node_counts = [g.node_count for g in graphs]
  assert sorted(node_counts) != node_counts

  # Test with a limited number of rows.
  graphs = list(graph_database.BufferedGraphReader(db, limit=5))
  assert len(graphs) == 5


def test_BufferedGraphDatabaseReader_next(db: graph_database.Database):
  """Test using next() to read from BufferedGraphReader()."""
  with db.Session(commit=True) as s:
    s.add(graph_database.GraphMeta(
          group = "train",
          bytecode_id = 1,
          source_name = 'foo',
          relpath = 'bar',
          language = 'c',
          node_count = 1,
          edge_count = 2,
          graph=graph_database.Graph(data=pickle.dumps({"a": 1}))))

  db_reader = graph_database.BufferedGraphReader(db)
  g = next(db_reader)
  assert g.group == "train"
  with pytest.raises(StopIteration):
    g = next(db_reader)


if __name__ == '__main__':
  test.Main()
