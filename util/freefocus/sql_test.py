"""Unit tests for //util/freefocus/sql.py"""
import pathlib
import sys
import tempfile
import typing

import pytest
from absl import app
from absl import flags

from util.freefocus import freefocus_pb2
from util.freefocus import sql


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> sql.Database:
  yield sql.Database(tempdir / 'test.db')


@pytest.fixture(scope='function')
def session(db) -> sql.Database.session_t:
  with db.Session() as session:
    yield session


def test_Person_CreateFromProto(session: sql.Database.session_t):
  """Short summary of test."""
  proto = freefocus_pb2.Person(
      id='cec',
      name=['Chris', 'Chris Cummins'],
      email=['foo@bar.com'],
      workspace_groups=[
        freefocus_pb2.Person.WorkspaceGroups(
            workspace_id='workspace', group_id=['cec', 'global'])
      ]
  )
  person = sql.Person.CreateFromProto(session, proto)
  print(person.ToProto())
  assert True


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
