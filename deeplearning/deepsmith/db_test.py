# Copyright (c) 2017-2020 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for :db."""
from deeplearning.deepsmith import db
from deeplearning.deepsmith import toolchain
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

pytest_plugins = ["deeplearning.deepsmith.tests.fixtures"]


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
  with test.Raises(NotImplementedError):
    db.Table.GetOrAdd("session", "proto")


def test_Table_ToProto_abstract():
  with test.Raises(NotImplementedError):
    db.Table().ToProto()


def test_Table_SetProto_abstract():
  with test.Raises(NotImplementedError):
    db.Table().SetProto("proto")


def test_Table_ProtoFromFile_abstract():
  with test.Raises(NotImplementedError):
    db.Table.ProtoFromFile("path")


def test_Table_FromFile_abstract():
  with test.Raises(NotImplementedError):
    db.Table.FromFile("session", "path")


def test_Table_abstract_methods():
  table = db.Table()
  with test.Raises(NotImplementedError):
    db.Table.GetOrAdd("session", "proto")
  with test.Raises(NotImplementedError):
    table.ToProto()
  with test.Raises(NotImplementedError):
    table.SetProto("proto")
  with test.Raises(NotImplementedError):
    db.Table.ProtoFromFile("path")
  with test.Raises(NotImplementedError):
    db.Table.FromFile("session", "path")


def test_Table_repr():
  string = str(db.Table())
  assert string == "TODO: Define Table.ToProto() method"


def test_StringTable_GetOrAdd_StringTooLongError(session):
  toolchain.Toolchain.GetOrAdd(session, "a" * toolchain.Toolchain.maxlen)
  with test.Raises(db.StringTooLongError):
    toolchain.Toolchain.GetOrAdd(
      session, "a" * (toolchain.Toolchain.maxlen + 1)
    )


def test_StringTable_TruncatedString(session):
  t = toolchain.Toolchain.GetOrAdd(session, "a" * 80)
  assert t.TruncatedString() == "a" * 80
  assert len(t.TruncatedString(n=70)) == 70
  assert t.TruncatedString(n=70) == "a" * 67 + "..."


def test_StringTable_TruncatedString_uninitialized():
  t = toolchain.Toolchain()
  assert len(t.TruncatedString()) == 0


def test_MakeEngine_unknown_backend():
  with test.Raises(NotImplementedError):
    db.MakeEngine(DataStoreProtoMock())


def test_MakeEngine_mysql_database_backtick():
  config = DataStoreProtoMock()
  config.HasField = lambda x: x == "mysql"
  config.mysql.database = "backtick`"
  with test.Raises(db.InvalidDatabaseConfig):
    db.MakeEngine(config)


def test_MakeEngine_postgresql_database_quote():
  config = DataStoreProtoMock()
  config.HasField = lambda x: x == "postgresql"
  config.postgresql.database = "singlequote'"
  with test.Raises(db.InvalidDatabaseConfig):
    db.MakeEngine(config)


if __name__ == "__main__":
  test.Main()
