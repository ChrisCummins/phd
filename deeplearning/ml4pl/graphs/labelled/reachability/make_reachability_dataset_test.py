"""Unit tests for //deeplearning/ml4pl/graphs/labelled/reachability:make_reachability_dataset."""
import pathlib

import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.reachability import \
  make_reachability_dataset as mrd

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:

  def _MakeLlvmBytecode(source_name) -> bytecode_database.LlvmBytecode:
    return bytecode_database.LlvmBytecode(
        source_name=source_name,
        relpath='bar.c',
        language='c',
        cflags='',
        charcount=0,
        linecount=0,
        bytecode="define i32 @A() #0 {\nret i32 10\n}",
        clang_returncode=0,
        error_message='',
        cfgs=[
            bytecode_database.ControlFlowGraphProto(
                cfg_id=0,
                status=0,
                proto='''\
name: "A"
block {
  name: "%1"
  text: "%2 = alloca i32, align 4\'nret void"
}
entry_block_index: 0
exit_block_index: 0\
''',
                error_message='',
                block_count=1,
                edge_count=0,
                is_strict_valid=False,
            ),
            bytecode_database.ControlFlowGraphProto(
                cfg_id=1,
                status=0,
                proto='''\
name: "A"
block {
  name: "%1"
  text: "%2 = alloca i32, align 4\\nret void"
}
entry_block_index: 0
exit_block_index: 0\
''',
                error_message='',
                block_count=1,
                edge_count=0,
                is_strict_valid=False,
            ),
        ])

  db = bytecode_database.Database(f'sqlite:///{tempdir}/bytecode_db')
  with db.Session(commit=True) as session:
    session.add_all([
        _MakeLlvmBytecode('poj-104:train'),
        _MakeLlvmBytecode('poj-104:val'),
        _MakeLlvmBytecode('poj-104:test'),
    ])
  return db


@pytest.fixture(scope='function')
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  return graph_database.Database(f'sqlite:///{tempdir}/graph_db')


def test_BytecodeExporter(bytecode_db: bytecode_database.Database,
                          graph_db: graph_database.Database):
  exporter = mrd.BytecodeExporter(bytecode_db, graph_db)

  with bytecode_db.Session() as s:
    ids = [r[0] for r in s.query(bytecode_database.LlvmBytecode.id)]

  exporter.ExportGroups({"a": ids})

  with graph_db.Session() as s:
    assert s.query(graph_database.GraphMeta).count() == 3


def test_ControlFlowGraphProtoExporter(bytecode_db: bytecode_database.Database,
                                       graph_db: graph_database.Database):
  exporter = mrd.ControlFlowGraphProtoExporter(bytecode_db, graph_db)

  with bytecode_db.Session() as s:
    ids = [r[0] for r in s.query(bytecode_database.LlvmBytecode.id)]

  exporter.ExportGroups({"a": ids})

  with graph_db.Session() as s:
    assert s.query(graph_database.GraphMeta).count() == 6


if __name__ == '__main__':
  test.Main()
