"""Pytest fixtures for me.db tests."""

import pathlib
import pytest
import tempfile
from absl import flags

from lib.labm8 import bazelutil
from util.me import me


FLAGS = flags.FLAGS
flags.DEFINE_string('integration_tests_inbox', None,
                    'If set, this sets the inbox path to be used by the '
                    'integration tests. This overrides the default in '
                    '//util/me/integration_tests/inbox.')

TEST_INBOX_PATH = bazelutil.DataPath('phd/util/me/tests/test_inbox')


@pytest.fixture(scope='function')
def mutable_db() -> me.Database:
  """Returns a populated database for the scope of the function."""
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    db = me.Database(pathlib.Path(d) / 'me.db')
    db.ImportMeasurementsFromInbox(TEST_INBOX_PATH)
    yield db


@pytest.fixture(scope='session')
def db() -> me.Database:
  """Returns a populated database that is reused for all tests.

  DO NOT MODIFY THE TEST DATABASE. This will break other tests. For a test that
  modifies the database, use the `mutable_db` fixture.
  """
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    db = me.Database(pathlib.Path(d) / 'me.db')
    db.ImportMeasurementsFromInbox(TEST_INBOX_PATH)
    yield db
