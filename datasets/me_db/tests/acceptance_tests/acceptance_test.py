# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Acceptance tests that perform dataset-agnostic, high level tests of a
database after importing an inbox.

These are implemented in a single file so that they can run in a single pytest
session. When running the integration tests with large datasets, the amount of
time taken to prepare the pytest fixtures becomes prohibitively expensive.

Usage:

    bazel test //datasets/me_db/tests/acceptance_tests --test_output=streamed
        --test_flag=--me_db_acceptance_tests_inbox=/path/to/inbox
"""
import datetime
import pathlib
import tempfile

import pandas as pd
import pytest
from sqlalchemy.sql.expression import func

from datasets.me_db import me_db
from datasets.me_db.tests.acceptance_tests import flags
from labm8.py import bazelutil
from labm8.py import test

FLAGS = flags.FLAGS

MODULE_UNDER_TEST = 'datasets.me_db'

TEST_INBOX_PATH = bazelutil.DataPath('phd/datasets/me_db/tests/test_inbox')

MILLISECONDS_IN_A_DAY = 1000 * 3600 * 24


def GetInboxPath() -> pathlib.Path:
  """Get the path of the inbox to populate a test database with."""
  # FLAGS.me_db_acceptance_tests_inbox is defined in :flags.py.
  #
  # A quirk in the combination of pytest and absl flags is that you can't define
  # a flag in the same file that you invoke pytest.main(). This is because the
  # pytest collector re-imports the file, causing absl to error because the
  # flags have already been defined.
  if FLAGS.me_db_acceptance_tests_inbox:
    return pathlib.Path(FLAGS.me_db_acceptance_tests_inbox)
  else:
    return TEST_INBOX_PATH


# Pytest fixtures cannot be moved out to a conftest.py file since they need to
# access FLAGS, and conftest loading occurs before app.RunWithArgs().
@pytest.fixture(scope='function')
def mutable_db() -> me_db.Database:
  """Returns a populated database for the scope of the function."""
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    db = me_db.Database(f'sqlite:///{d}/me.db')
    db.ImportMeasurementsFromInboxImporters(GetInboxPath())
    yield db


@pytest.fixture(scope='session')
def db() -> me_db.Database:
  """Returns a populated database that is reused for all tests.

  DO NOT MODIFY THE TEST DATABASE. This will break other tests. For a test that
  modifies the database, use the `mutable_db` fixture.
  """
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    db = me_db.Database(f'sqlite:///{d}/me.db')
    db.ImportMeasurementsFromInboxImporters(GetInboxPath())
    yield db


def test_num_measurements(db: me_db.Database):
  """Test that at least measurement has been imported."""
  with db.Session() as s:
    q = s.query(me_db.Measurement)

    assert q.count() > 1


def test_groups_not_empty(db: me_db.Database):
  """Test the number of measurements."""
  with db.Session() as s:
    q = s.query(me_db.Measurement) \
      .filter(me_db.Measurement.group == '')

    # There should be no missing "group" field. Instead of an empty group, use
    # the value "default".
    assert q.count() == 0


def test_no_dates_in_the_future(db: me_db.Database):
  """Test that there are no dates in the future."""
  now = datetime.datetime.now()
  with db.Session() as s:
    q = s.query(me_db.Measurement) \
      .filter(me_db.Measurement.date > now)

    assert q.count() == 0


def test_life_cycle_daily_total(db: me_db.Database):
  """Test that the sum of all measurements for a day is <= 24 hours."""
  with db.Session() as s:
    q = s.query(func.DATE(me_db.Measurement.date).label('date'),
                (func.sum(me_db.Measurement.value)).label('time')) \
      .filter(me_db.Measurement.source == 'LifeCycle') \
      .group_by(me_db.Measurement.date)
    df = pd.read_sql(q.statement, q.session.bind)

    assert df[df.time > MILLISECONDS_IN_A_DAY].empty


def test_life_cycle_dates_do_not_overflow(db: me_db.Database):
  """Test that no LifeCycle measurements overflow to the next day."""
  with db.Session() as s:
    q = s.query(me_db.Measurement.date, me_db.Measurement.value) \
      .filter(me_db.Measurement.source == 'LifeCycle')

    for start_date, value in q.distinct():
      end_date = start_date + datetime.timedelta(milliseconds=value)
      # The number of days between the end and start of the measurement.
      day_diff = (end_date - start_date).days
      if not day_diff:
        continue
      # The only case where the end date is allowed to differ from the start
      # date is when we have overflowed to midnight (00:00:00) the next day.
      if not (day_diff == 1 and end_date.hour == 0 and end_date.minute == 0 and
              end_date.second == 0):
        pytest.fail(
            f'Date {start_date} overflows when adding measurement {value} ms '
            f'(calculated end date: {end_date})')


def test_life_cycle_dates_are_unique(db: me_db.Database):
  """There can be no duplicate dates in Life Cycle measurements."""
  with db.Session() as s:
    q = s.query(me_db.Measurement.date) \
      .filter(me_db.Measurement.source == 'LifeCycle')
    num_life_cycle_dates = q.count()

    q = s.query(me_db.Measurement.date) \
      .filter(me_db.Measurement.source == 'LifeCycle') \
      .distinct()
    num_distinct_life_cycle_dates = q.count()

    # There are no duplicate start dates.
    assert num_distinct_life_cycle_dates == num_life_cycle_dates


if __name__ == '__main__':
  test.Main()
