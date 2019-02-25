"""Unit tests for :db."""

import pytest
from absl import flags

from deeplearning.deepsmith import db
from deeplearning.deepsmith import toolchain
from labm8 import test

FLAGS = flags.FLAGS


def HasFieldMock(self, name):
  """Mock for proto 'HasField' method"""
  del self, name
  return False


class DataStoreProtoMock(object):
  """DataStore proto mock class."""
  testonly = True

  HasField = HasFieldMock

  class DatabaseMock(object):
    """Database config mock class."""
    HasField = HasFieldMock

  mysql = DatabaseMock()
  postgresql = DatabaseMock()
  sqlite = DatabaseMock()


def test_Table_GetOrAdd_abstract():
  with pytest.raises(NotImplementedError):
    db.Table.GetOrAdd('session', 'proto')


def test_Table_ToProto_abstract():
  with pytest.raises(NotImplementedError):
    db.Table().ToProto()


def test_Table_SetProto_abstract():
  with pytest.raises(NotImplementedError):
    db.Table().SetProto('proto')


def test_Table_ProtoFromFile_abstract():
  with pytest.raises(NotImplementedError):
    db.Table.ProtoFromFile('path')


def test_Table_FromFile_abstract():
  with pytest.raises(NotImplementedError):
    db.Table.FromFile('session', 'path')


def test_Table_abstract_methods():
  table = db.Table()
  with pytest.raises(NotImplementedError):
    db.Table.GetOrAdd('session', 'proto')
  with pytest.raises(NotImplementedError):
    table.ToProto()
  with pytest.raises(NotImplementedError):
    table.SetProto('proto')
  with pytest.raises(NotImplementedError):
    db.Table.ProtoFromFile('path')
  with pytest.raises(NotImplementedError):
    db.Table.FromFile('session', 'path')


def test_Table_repr():
  string = str(db.Table())
  assert string == 'TODO: Define Table.ToProto() method'


def test_StringTable_GetOrAdd_StringTooLongError(session):
  toolchain.Toolchain.GetOrAdd(session, 'a' * toolchain.Toolchain.maxlen)
  with pytest.raises(db.StringTooLongError):
    toolchain.Toolchain.GetOrAdd(session,
                                 'a' * (toolchain.Toolchain.maxlen + 1))


def test_StringTable_TruncatedString(session):
  t = toolchain.Toolchain.GetOrAdd(session, 'a' * 80)
  assert t.TruncatedString() == 'a' * 80
  assert len(t.TruncatedString(n=70)) == 70
  assert t.TruncatedString(n=70) == 'a' * 67 + '...'


def test_StringTable_TruncatedString_uninitialized():
  t = toolchain.Toolchain()
  assert len(t.TruncatedString()) == 0


def test_MakeEngine_unknown_backend():
  with pytest.raises(NotImplementedError):
    db.MakeEngine(DataStoreProtoMock())


def test_MakeEngine_mysql_database_backtick():
  config = DataStoreProtoMock()
  config.HasField = lambda x: x == 'mysql'
  config.mysql.database = 'backtick`'
  with pytest.raises(db.InvalidDatabaseConfig):
    db.MakeEngine(config)


def test_MakeEngine_postgresql_database_quote():
  config = DataStoreProtoMock()
  config.HasField = lambda x: x == 'postgresql'
  config.postgresql.database = "singlequote'"
  with pytest.raises(db.InvalidDatabaseConfig):
    db.MakeEngine(config)


if __name__ == '__main__':
  test.Main()
