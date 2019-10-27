"""Unit tests for //deeplearning/ml4pl/graphs/labelled/classifyapp:make_classifyapp_dataset."""
import pathlib
import pickle
import typing

import networkx as nx
import numpy as np
import pytest
from labm8 import app
from labm8 import bazelutil
from labm8 import fs
from labm8 import test

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.classifyapp import \
  make_classifyapp_dataset as classifyapp

FLAGS = app.FLAGS

# Real-world bytecodes that can be converted to XFGs.
REAL_OKAY_BYTECODES = [
    fs.Read(
        bazelutil.DataPath(
            'phd/deeplearning/ml4pl/graphs/labelled/classifyapp/test_data/bytecode.ll'
        )),
]

# Real-world bytecodes that fail to be converted to XFGs.
REAL_FAIL_BYTECODES = [
    fs.Read(
        bazelutil.DataPath(
            'phd/deeplearning/ml4pl/graphs/labelled/classifyapp/test_data/164689.ll'
        )),
]

INST2VEC_DICITONARY_PATH = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/dic_pickle')


@pytest.fixture(scope='function')
def bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:
  """Return a bytecode database with three bytecodes."""

  def _MakeLlvmBytecode(source_name, relpath) -> bytecode_database.LlvmBytecode:
    return bytecode_database.LlvmBytecode(
        source_name=source_name,
        relpath=relpath,
        language='c',
        cflags='',
        charcount=0,
        linecount=0,
        bytecode="define i32 @A() #0 {\nret i32 10\n}",
        clang_returncode=0,
        error_message='',
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


def _Pickle(path: pathlib.Path, data: typing.Any) -> str:
  with open(path, 'wb') as f:
    pickle.dump(data, f)
  return str(path)


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

  classifyapp.AddXfgFeatures(graph, {
      'foo': 1,
      'bar': 2,
      "!UNK": -1,
  })

  label_one_hot = np.zeros(104, dtype=np.int32)
  label_one_hot[12] = 1
  assert np.array_equal(graph.y, label_one_hot)
  assert np.array_equal(graph.edges['A', 'B', 0]['x'], [1])
  assert np.array_equal(graph.edges['B', 'C', 0]['x'], [2])
  assert np.array_equal(graph.edges['C', 'D', 0]['x'], [-1])


def test_BytecodeExporter_num_rows(bytecode_db: bytecode_database.Database,
                                   graph_db: graph_database.Database,
                                   tempdir: pathlib.Path):
  """Test the number of bytecodes exported."""
  FLAGS.dictionary = _Pickle(tempdir / 'dic', {
      'foo': 1,
      'bar': 2,
      "!UNK": -1,
  })

  exporter = classifyapp.Exporter(bytecode_db, graph_db)
  exporter.Export()

  with graph_db.Session() as s:
    assert s.query(graph_database.GraphMeta).count() == 3


def test_BytecodeExporter_graph_dict(bytecode_db: bytecode_database.Database,
                                     graph_db: graph_database.Database,
                                     tempdir: pathlib.Path):
  """Test the graph_dict properties of exported bytecodes."""
  FLAGS.dictionary = _Pickle(tempdir / 'dic', {
      'foo': 1,
      'bar': 2,
      "!UNK": -1,
  })

  exporter = classifyapp.Exporter(bytecode_db, graph_db)
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

    assert graph_dict['edge_x'].shape == (gm.edge_type_count,)
    assert graph_dict['graph_y'].shape == (104,)


@pytest.mark.parametrize('bytecode_id', range(len(REAL_OKAY_BYTECODES)))
def test_BytecodeExporter_okay_bytecodes(
    tempdir: pathlib.Path, graph_db: graph_database.Database, bytecode_id: int):
  """Create creating Graphs from single-entry databases of real bytecodes."""
  bytecode = REAL_OKAY_BYTECODES[bytecode_id]

  db = bytecode_database.Database(f'sqlite:///{tempdir}/bytecode_db')
  with db.Session(commit=True) as session:
    session.add(
        bytecode_database.LlvmBytecode(
            source_name='poj-104:train',
            relpath='1/bytecode.ll',
            language='cpp',
            cflags='',
            charcount=len(bytecode),
            linecount=len(bytecode.split('\n')),
            bytecode=bytecode,
            clang_returncode=0,
            error_message='',
        ))
  exporter = classifyapp.Exporter(db, graph_db, INST2VEC_DICTIONARY)
  exporter.Export()

  with graph_db.Session() as session:
    assert session.query(graph_database.GraphMeta).count() == 1


@pytest.mark.xfail()
@pytest.mark.parametrize('bytecode_id', range(len(REAL_FAIL_BYTECODES)))
def test_BytecodeExporter_okay_bytecodes(
    tempdir: pathlib.Path, graph_db: graph_database.Database, bytecode_id: int):
  """Create creating Graphs from single-entry databases of real bytecodes."""
  FLAGS.dictionary = INST2VEC_DICITONARY_PATH

  bytecode = REAL_FAIL_BYTECODES[bytecode_id]

  db = bytecode_database.Database(f'sqlite:///{tempdir}/bytecode_db')
  with db.Session(commit=True) as session:
    session.add(
        bytecode_database.LlvmBytecode(
            source_name='poj-104:train',
            relpath='1/bytecode.ll',
            language='cpp',
            cflags='',
            charcount=len(bytecode),
            linecount=len(bytecode.split('\n')),
            bytecode=bytecode,
            clang_returncode=0,
            error_message='',
        ))
  exporter = classifyapp.Exporter(db, graph_db)
  exporter.Export()


if __name__ == '__main__':
  test.Main()
