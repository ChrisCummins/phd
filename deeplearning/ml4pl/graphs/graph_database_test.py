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


if __name__ == '__main__':
  test.Main()
