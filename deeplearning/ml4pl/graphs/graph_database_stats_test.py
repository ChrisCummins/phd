"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
import numpy as np
import pathlib
import pickle
import pytest

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_stats as stats
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> graph_database.Database:
  """Fixture that returns an sqlite database."""
  yield graph_database.Database(f'sqlite:///{tempdir}/db')


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
        node_features_dimensionality=2,
        graph_labels_dimensionality=1,
        graph=graph_database.Graph(data=pickle.dumps(
            {
                "node_x": np.array(np.array([1, 2], dtype=np.int32)),
                "graph_y": np.array([1.4], dtype=np.float32)
            })))

  with db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  return db


def test_GraphDatabaseStats_graph_count(db_512: graph_database.Database):
  """Test that the expected number of graphs are returned"""
  s = stats.GraphDatabaseStats(db_512)
  assert s.graph_count == 512


def test_GraphDatabaseStats_repr(db_512: graph_database.Database):
  """Test the string representation of the stats object"""
  s = stats.GraphDatabaseStats(db_512)
  assert str(s) == ("Graphs database: 512 instances, 1 edge type, "
                    "2-d node features, 1-d graph labels, "
                    "max 511 nodes, max 2 edges")


def test_GraphDictDatabaseStats_repr(db_512: graph_database.Database):
  """Test the string representation of the stats object"""
  s = stats.GraphDictDatabaseStats(db_512)
  assert str(s) == ("Graphs database: 512 instances, 1 edge type, "
                    "2-d int32 node features, 1-d float32 graph labels, "
                    "max 511 nodes, max 2 edges")


if __name__ == '__main__':
  test.Main()
