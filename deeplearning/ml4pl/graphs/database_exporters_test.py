"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
import pathlib
import pickle
import typing

import pytest

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import database_exporters
from deeplearning.ml4pl.graphs import graph_database
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:
  """Fixture that returns an sqlite database."""
  yield bytecode_database.Database(f'sqlite:///{tempdir}/bytecode.db')


@pytest.fixture(scope='function')
def bytecode_db_512(
    bytecode_db: bytecode_database.Database) -> bytecode_database.Database:
  """Fixture that returns an sqlite database."""
  with bytecode_db.Session(commit=True) as session:
    session.add_all([
        bytecode_database.LlvmBytecode(
            source_name='foo',
            relpath=f'bar_{i}.c',
            language='c',
            cflags='',
            charcount=i,
            linecount=10,
            bytecode=f'bytecode_{i}',
            clang_returncode=0,
            error_message='',
            bytecode_sha1='0xdeadbeef',
        ) for i in range(512)
    ])
  yield bytecode_db


@pytest.fixture(scope='function')
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  """Fixture that returns an sqlite database."""
  yield graph_database.Database(f'sqlite:///{tempdir}/graphs.db')


@pytest.fixture(scope='function')
def graph_db_512(graph_db: graph_database.Database) -> graph_database.Database:
  """Fixture that returns an sqlite database."""
  with graph_db.Session(commit=True) as session:
    session.add_all([
        graph_database.GraphMeta(
            group="train",
            bytecode_id=i,
            source_name='foo',
            relpath='bar',
            language='c',
            node_count=1,
            edge_count=2,
            edge_position_max=0,
            loop_connectedness=0,
            undirected_diameter=0,
            graph=graph_database.Graph(pickled_data=pickle.dumps({
                "a": 1,
                "b": 2
            }))) for i in range(512)
    ])
  yield graph_db


def _MockProcessBytecodeInputs(bytecode_db: bytecode_database.Database,
                               bytecode_ids: typing.List[int]):
  """A fake bytecode job creator returns three graphs per input."""
  with bytecode_db.Session() as session:
    bytecodes = session.query(bytecode_database.LlvmBytecode.id,
                                     bytecode_database.LlvmBytecode.bytecode) \
      .filter(bytecode_database.LlvmBytecode.id.in_(bytecode_ids)) \
      .all()
  bytecode_db.Close()  # Don't leave the database connection lying around.
  return [
      graph_database.GraphMeta(
          bytecode_id=bytecode_id,
          source_name='foo',
          relpath='bar',
          language='c',
          node_count=1,
          edge_count=2,
          edge_position_max=0,
          loop_connectedness=0,
          undirected_diameter=0,
          graph=graph_database.Graph(pickled_data=pickle.dumps({
              "id": bytecode_id,
              "i": i,
          }))) for i, (bytecode_id, bytecode) in enumerate(bytecodes)
  ]


class MockBytecodeExporter(database_exporters.BytecodeDatabaseExporterBase):
  """A mock bytecode exporter"""

  def GetProcessInputs(self):
    return _MockProcessBytecodeInputs


def test_MockBytecodeExporter(bytecode_db_512: bytecode_database.Database,
                              graph_db: graph_database.Database):
  MockBytecodeExporter()(bytecode_db_512, [graph_db])

  with graph_db.Session() as s:
    graphs = s.query(graph_database.GraphMeta).all()

  assert len(graphs) == 512

  for graph in graphs:
    assert graph.group in {'train', 'val', 'test'}
    assert graph.bytecode_id <= 512


def _MockProcessGraphInputs(graph_db_session: graph_database.Database.Session,
                            bytecode_ids: typing.List[int]):
  """A fake bytecode job creator returns three graphs per input."""
  graph_inputs = graph_db_session.query(graph_database.GraphMeta) \
    .filter(graph_database.GraphMeta.id.in_(bytecode_ids))
  return [
      graph_database.GraphMeta(bytecode_id=graph_meta.bytecode_id,
                               source_name='foo',
                               relpath='bar',
                               language='c',
                               node_count=1,
                               edge_count=2,
                               edge_position_max=0,
                               loop_connectedness=0,
                               undirected_diameter=0,
                               graph=graph_database.Graph(
                                   pickled_data=pickle.dumps({
                                       "id": graph_meta.bytecode_id,
                                       "i": i,
                                   })))
      for i, graph_meta in enumerate(graph_inputs)
  ]


class MockGraphExporter(database_exporters.GraphDatabaseExporterBase):
  """A mock bytecode exporter"""

  def GetProcessInputs(self):
    return _MockProcessGraphInputs


def test_MockGraphExporter(graph_db_512: bytecode_database.Database,
                           graph_db: graph_database.Database):
  MockGraphExporter()(graph_db_512, [graph_db])

  with graph_db.Session() as s:
    graphs = s.query(graph_database.GraphMeta).all()

  assert len(graphs) == 512

  for graph in graphs:
    assert graph.group in {'train', 'val', 'test'}
    assert graph.bytecode_id <= 512


if __name__ == '__main__':
  test.Main()
