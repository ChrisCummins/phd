"""Unit tests for //deeplearning/ml4pl/graphs/labelled/classifyapp:make_classifyapp_dataset."""
import networkx as nx
import numpy as np
import pathlib
import pickle
import pytest

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.classifyapp import \
  make_classifyapp_dataset as classifyapp
from labm8 import app
from labm8 import bazelutil
from labm8 import fs
from labm8 import test


FLAGS = app.FLAGS

REAL_BYTECODE = fs.Read(bazelutil.DataPath(
    'phd/deeplearning/ml4pl/graphs/labelled/classifyapp/test_data/bytecode.ll'))

INST2VEC_DICITONARY_PATH = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/dic_pickle')

with open(INST2VEC_DICITONARY_PATH, 'rb') as f:
  INST2VEC_DICTIONARY = pickle.load(f)


@pytest.fixture(scope='function')
def bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:
  """Return a bytecode database with three bytecodes."""

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

def test_AddXfgFeatures_counters():
  """Test that graph annotations are added."""
  graph = nx.MultiDiGraph()
  graph.relpath = '13/foo.ll'
  graph.add_edge('A', 'B', flow='ctrl', stmt='foo')
  graph.add_edge('B', 'C', flow='none', stmt='bar')
  graph.add_edge('C', 'D', flow='data', stmt='car')

  stmt_count, unknown_count = classifyapp.AddXfgFeatures(
      graph, {
        'foo': 1,
        'bar': 2,
        "!UNK": -1,
      })

  assert stmt_count == 3
  assert unknown_count == 1

def test_AddXfgFeatures_feature_values():
  """Test that graph annotations are added."""
  graph = nx.MultiDiGraph()
  graph.relpath = '13/foo.ll'
  graph.add_edge('A', 'B', flow='ctrl', stmt='foo')
  graph.add_edge('B', 'C', flow='none', stmt='bar')
  graph.add_edge('C', 'D', flow='data', stmt='car')

  classifyapp.AddXfgFeatures(
      graph, {
        'foo': 1,
        'bar': 2,
        "!UNK": -1,
      })

  assert np.array_equal(graph.y, [13])
  assert np.array_equal(graph.edges['A', 'B', 0]['x'], [1])
  assert np.array_equal(graph.edges['B', 'C', 0]['x'], [2])
  assert np.array_equal(graph.edges['C', 'D', 0]['x'], [-1])


def test_BytecodeExporter_num_rows(bytecode_db: bytecode_database.Database,
                          graph_db: graph_database.Database):
  """Test the number of bytecodes exported."""
  exporter = classifyapp.Exporter(bytecode_db, graph_db, {
    '!UNK': 1,
  })
  exporter.Export()

  with graph_db.Session() as s:
    assert s.query(graph_database.GraphMeta).count() == 3


def test_BytecodeExporter_graph_dict(bytecode_db: bytecode_database.Database,
                                     graph_db: graph_database.Database):
  """Test the graph_dict properties of exported bytecodes."""
  exporter = classifyapp.Exporter(bytecode_db, graph_db, {
    '!UNK': 1,
  })
  exporter.Export()

  with graph_db.Session() as s:
    assert s.query(graph_database.GraphMeta).count() == 3
    gm = s.query(graph_database.GraphMeta).first()
    graph_dict = gm.pickled_data

    assert 'node_x' not in graph_dict
    assert 'node_y' not in graph_dict
    assert 'edge_x' in graph_dict
    assert 'edge_y' not in graph_dict
    assert 'graph_x' not in graph_dict
    assert 'graph_y' in graph_dict


@pytest.fixture(scope='function')
def poj104_bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:
  """Return an bytecode database with a single bytecode from POJ-104 dataset."""
  db = bytecode_database.Database(f'sqlite:///{tempdir}/bytecode_db')
  with db.Session(commit=True) as session:
    session.add(bytecode_database.LlvmBytecode(
        source_name='poj-104:train',
        relpath = '1/bytecode.ll',
        language = 'cpp',
        cflags = '',
        charcount = len(REAL_BYTECODE),
        linecount = len(REAL_BYTECODE.split('\n')),
        bytecode = REAL_BYTECODE,
        clang_returncode = 0,
        error_message = '',
    ))
  return db


def test_BytecodeExporter_poj104_smoke_test(
    poj104_bytecode_db: bytecode_database.Database,
    graph_db: graph_database.Database):
  exporter = classifyapp.Exporter(
      poj104_bytecode_db, graph_db, INST2VEC_DICTIONARY)
  exporter.Export()

  with graph_db.Session() as session:
    assert session.query(graph_database.GraphMeta).count() == 1



if __name__ == '__main__':
  test.Main()
