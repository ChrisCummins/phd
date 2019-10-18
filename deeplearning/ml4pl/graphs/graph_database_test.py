"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
import networkx as nx
import pathlib
import pickle
import pytest

from deeplearning.ml4pl.graphs import graph_database
from labm8 import app
from labm8 import test

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
def db_512(db: graph_database.Database) -> graph_database.Database:
  """Fixture which returns a database with 512 graphs, indexed by node_count."""

  def _MakeGraphMeta(i):
    return graph_database.GraphMeta(
        group="train",
        bytecode_id=1,
        source_name='foo',
        relpath='bar',
        language='c',
        node_count=i,
        edge_count=2,
        graph=graph_database.Graph(data=pickle.dumps({"a": 1})))

  with db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  return db


@pytest.mark.parametrize('buffer_size', [1, 10, 25, 10000])
def test_BufferedGraphReader_length(db_512: graph_database.Database,
                                    buffer_size: int):
  """Test that the expected number of graphs are returned"""
  graphs = list(
      graph_database.BufferedGraphReader(db_512, buffer_size=buffer_size))
  assert len(graphs) == 512
  assert all([g.bytecode_id == 1 for g in graphs])
  assert all([g.node_count == i for i, g in enumerate(graphs)])


def test_BufferedGraphReader_filter(db_512: graph_database.Database):
  """Test using a filter callback."""
  filter_cb = lambda: graph_database.GraphMeta.node_count % 2 == 0
  graphs = list(
      graph_database.BufferedGraphReader(db_512,
                                         filters=[filter_cb],
                                         buffer_size=10))
  assert len(graphs) == 256


def test_BufferedGraphReader_filters(db_512: graph_database.Database):
  """Test using multiple filters in combination."""
  filters = [
      lambda: graph_database.GraphMeta.node_count % 2 == 0, lambda:
      graph_database.GraphMeta.id < 256
  ]
  graphs = list(graph_database.BufferedGraphReader(db_512, filters=filters))
  assert len(graphs) == 128


def test_BufferedGraphReader_order_by_random(db_512: graph_database.Database):
  """Test using `order_by_random` arg to randomize row order."""
  graphs = list(graph_database.BufferedGraphReader(db_512,
                                                   order_by_random=True))
  node_counts = [g.node_count for g in graphs]
  # Flaky: there is a possibility that random order returns all rows in order!
  assert sorted(node_counts) != node_counts


def test_BufferedGraphReader_limit(db_512: graph_database.Database):
  """Test using `limit` arg to limit number of returned rows."""
  graphs = list(graph_database.BufferedGraphReader(db_512, limit=5))
  assert len(graphs) == 5


def test_BufferedGraphReader_next(db_512: graph_database.Database):
  """Test using next() to read from BufferedGraphReader()."""
  db_reader = graph_database.BufferedGraphReader(db_512)
  for _ in range(512):
    next(db_reader)
  with pytest.raises(StopIteration):
    next(db_reader)


if __name__ == '__main__':
  test.Main()
