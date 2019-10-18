"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
import pathlib
import pickle
import pytest

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as reader
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


@pytest.mark.parametrize('buffer_size', [1, 10, 25, 10000])
def test_BufferedGraphReader_length(db_512: graph_database.Database,
                                    buffer_size: int):
  """Test that the expected number of graphs are returned"""
  graphs = list(
      reader.BufferedGraphReader(db_512, buffer_size=buffer_size))
  assert len(graphs) == 512
  assert all([g.bytecode_id == 1 for g in graphs])
  assert all([g.node_count == i for i, g in enumerate(graphs)])


@pytest.mark.parametrize('buffer_size', [1, 10, 25, 10000])
def test_BufferedGraphReader_filter(db_512: graph_database.Database,
                                    buffer_size: int):
  """Test using a filter callback."""
  filter_cb = lambda: graph_database.GraphMeta.node_count % 2 == 0
  graphs = list(
      reader.BufferedGraphReader(db_512,
                                         filters=[filter_cb],
                                         buffer_size=10))
  assert len(graphs) == 256


@pytest.mark.parametrize('buffer_size', [1, 10, 25, 10000])
def test_BufferedGraphReader_filters(db_512: graph_database.Database,
                                     buffer_size: int):
  """Test using multiple filters in combination."""
  filters = [
    lambda: graph_database.GraphMeta.node_count % 2 == 0, lambda:
    graph_database.GraphMeta.id < 256
  ]
  graphs = list(reader.BufferedGraphReader(db_512, filters=filters))
  assert len(graphs) == 128


@pytest.mark.parametrize('buffer_size', [1, 10, 25, 10000])
def test_BufferedGraphReader_order_by_random(db_512: graph_database.Database,
                                             buffer_size: int):
  """Test using `order_by_random` arg to randomize row order."""
  graphs = list(reader.BufferedGraphReader(db_512,
                                                   order_by_random=True))
  node_counts = [g.node_count for g in graphs]
  # Flaky: there is a possibility that random order returns all rows in order!
  assert sorted(node_counts) != node_counts


@pytest.mark.parametrize('buffer_size', [1, 10, 25, 10000])
def test_BufferedGraphReader_limit(db_512: graph_database.Database,
                                   buffer_size: int):
  """Test using `limit` arg to limit number of returned rows."""
  graphs = list(reader.BufferedGraphReader(db_512, limit=5))
  assert len(graphs) == 5


@pytest.mark.parametrize('buffer_size', [1, 10, 25, 10000])
def test_BufferedGraphReader_next(db_512: graph_database.Database,
                                  buffer_size: int):
  """Test using next() to read from BufferedGraphReader()."""
  db_reader = reader.BufferedGraphReader(db_512)
  for _ in range(512):
    next(db_reader)
  with pytest.raises(StopIteration):
    next(db_reader)


if __name__ == '__main__':
  test.Main()
