"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
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
        graph=graph_database.Graph(data=pickle.dumps({"a": 1})))

  with db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  return db


def test_GraphDatabaseStats_graph_count(db_512: graph_database.Database):
  """Test that the expected number of graphs are returned"""
  s = stats.GraphDatabaseStats(db_512)
  assert s.graph_count == 512


if __name__ == '__main__':
  test.Main()
