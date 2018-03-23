"""Unit tests for :db."""
import sys

import pytest
from absl import app

from deeplearning.deepsmith import db
from deeplearning.deepsmith import toolchain


class DataStoreProtoMock(object):
  """DataStore proto mock class."""
  testonly = True

  def HasField(self, name):
    return False


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
    toolchain.Toolchain.GetOrAdd(
        session, 'a' * (toolchain.Toolchain.maxlen + 1))


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


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
