"""Unit tests for //deeplearning/ml4pl/bytecode:splitters."""
import pathlib

import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.bytecode import splitters

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db_512(tempdir: pathlib.Path) -> bytecode_database.Database:
  db = bytecode_database.Database(f'sqlite:///{tempdir}/db')
  with db.Session(commit=True) as session:
    session.add_all([
        bytecode_database.LlvmBytecode(
            source_name='foo',
            relpath='bar.c',
            language='c',
            cflags='',
            charcount=0,
            linecount=0,
            bytecode='',
            clang_returncode=0,
            error_message='',
        ) for _ in range(512)
    ])
  return db


def test_GetTrainValTestGroups_group_names(db_512: bytecode_database.Database):
  groups = splitters.GetTrainValTestGroups(db_512)
  assert set(groups.keys()) == {'train', 'val', 'test'}


def test_GetTrainValTestGroups_group_sizes(db_512: bytecode_database.Database):
  groups = splitters.GetTrainValTestGroups(db_512)
  assert sum(len(v) for v in groups.values()) == 512


def test_GetTrainValTestGroups_group_ratios(db_512: bytecode_database.Database):
  groups = splitters.GetTrainValTestGroups(db_512,
                                           train_val_test_ratio=[2, 1, 1])
  assert len(groups['train']) == len(groups['val']) * 2
  assert len(groups['train']) == len(groups['test']) * 2


def test_GetTrainValTestGroups_unique_ids(db_512: bytecode_database.Database):
  groups = splitters.GetTrainValTestGroups(db_512,
                                           train_val_test_ratio=[2, 1, 1])
  assert len(set(groups['train'] + groups['val'] + groups['test'])) == 512


@pytest.fixture(scope='function')
def db_poj104(tempdir: pathlib.Path) -> bytecode_database.Database:

  def _MakeLlvmBytecode(source_name) -> bytecode_database.LlvmBytecode:
    return bytecode_database.LlvmBytecode(
        source_name=source_name,
        relpath='bar.c',
        language='c',
        cflags='',
        charcount=0,
        linecount=0,
        bytecode='',
        clang_returncode=0,
        error_message='',
    )

  db = bytecode_database.Database(f'sqlite:///{tempdir}/db')
  with db.Session(commit=True) as session:
    session.add_all([
        _MakeLlvmBytecode('poj-104:train'),
        _MakeLlvmBytecode('poj-104:val'),
        _MakeLlvmBytecode('poj-104:test'),
    ])
  return db


def test_GetPoj104BytecodeGroups_group_names(
    db_poj104: bytecode_database.Database):
  groups = splitters.GetPoj104BytecodeGroups(db_poj104)
  assert set(groups.keys()) == {'train', 'val', 'test'}


def test_GetPoj104BytecodeGroups_group_sizes(
    db_poj104: bytecode_database.Database):
  groups = splitters.GetPoj104BytecodeGroups(db_poj104)
  assert len(groups['train']) == 1
  assert len(groups['val']) == 1
  assert len(groups['test']) == 1


def test_GetPoj104BytecodeGroups_unique_ids(
    db_poj104: bytecode_database.Database):
  groups = splitters.GetPoj104BytecodeGroups(db_poj104)
  assert len(set(groups['train'] + groups['val'] + groups['test'])) == 3


def test_GetGroupsFromFlags(db_512: bytecode_database.Database):
  splitters.GetGroupsFromFlags(db_512)


if __name__ == '__main__':
  test.Main()
