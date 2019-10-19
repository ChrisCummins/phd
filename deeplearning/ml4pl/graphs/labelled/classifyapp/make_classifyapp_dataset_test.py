"""Unit tests for //deeplearning/ml4pl/graphs/labelled/classifyapp:make_classifyapp_dataset."""
import networkx as nx
import pathlib
import pytest

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.classifyapp import \
  make_classifyapp_dataset as classifyapp
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:

  def _MakeLlvmBytecode(source_name, relpath) -> bytecode_database.LlvmBytecode:
    return bytecode_database.LlvmBytecode(
        source_name=source_name,
        relpath = relpath,
        language = 'c',
        cflags = '',
        charcount = 0,
        linecount = 0,
        bytecode = "define i32 @A() #0 {\nret i32 10\n}",
        clang_returncode = 0,
        error_message = '',
    )

  db = bytecode_database.Database(f'sqlite:///{tempdir}/bytecode_db')
  with db.Session(commit=True) as session:
    session.add_all([
      _MakeLlvmBytecode('poj-104:train', '1/foo.ll'),
      _MakeLlvmBytecode('poj-104:val', '2/bar.ll'),
      _MakeLlvmBytecode('poj-104:test', '1/foo.ll'),
    ])
  return db


@pytest.fixture(scope='function')
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  return graph_database.Database(f'sqlite:///{tempdir}/graph_db')


def test_GetGraphLabel():
  assert classifyapp.GetGraphLabel('1/bar.ll') == 1
  assert classifyapp.GetGraphLabel('2/foo.ll') == 2

def test_AddGraphFeatures():
  graph = nx.MultiDiGraph()
  graph.relapth('13/foo.ll')
  graph.add_edge('A', 'B', flow='ctrl', stmt='foo')
  graph.add_edge('B', 'C', flow='none', stmt='bar')
  graph.add_edge('C', 'D', flow='data', stmt='car')

  classifyapp.AddGraphFeatures(
      graph, {
        'foo': 1,
        'bar': 2,
        "!UNK": -1,
      })

  assert np.array_equals(graph.y, [13])
  assert np.array_equals(graph.edges['A', 'B', 0]['x'], [1])
  assert np.array_equals(graph.edges['B', 'C', 0]['x'], [2])
  assert np.array_equals(graph.edges['C', 'D', 0]['x'], [-1])


def test_BytecodeExporter(bytecode_db: bytecode_database.Database,
                          graph_db: graph_database.Database):
  exporter = classifyapp.Exporter(bytecode_db, graph_db)

  with bytecode_db.Session() as s:
    ids = [r[0] for r in s.query(bytecode_database.LlvmBytecode.id)]

  exporter.Export({
    '!UNK': 1,
  })

  with graph_db.Session() as s:
    assert s.query(graph_database.GraphMeta).count() == 3


if __name__ == '__main__':
  test.Main()
