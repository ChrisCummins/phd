"""Unit tests for //util/freefocus/sql.py"""
import pathlib

import pytest

from labm8 import app
from labm8 import sqlutil
from labm8 import test
from util.freefocus import freefocus_pb2
from util.freefocus import sql


FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> sql.Database:
  yield sql.Database(tempdir / 'test.db')


@pytest.fixture(scope='function')
def session(db) -> sqlutil.Session:
  with db.Session() as session:
    yield session


def test_Person_CreateFromProto(session: sqlutil.Session):
  """Short summary of test."""
  proto = freefocus_pb2.Person(
      id='cec',
      name=['Chris', 'Chris Cummins'],
      email=['foo@bar.com'],
      workspace_groups=[
          freefocus_pb2.Person.WorkspaceGroups(
              workspace_id='workspace', group_id=['cec', 'global'])
      ])
  person = sql.Person.CreateFromProto(session, proto)
  print(person.ToProto())
  assert True


if __name__ == '__main__':
  test.Main()
