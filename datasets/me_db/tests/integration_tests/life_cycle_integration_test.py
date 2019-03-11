"""Integration tests for Life Cycle import to me_db."""
import datetime

from datasets.me_db import me_db
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_measurements_count(db: me_db.Database):
  """Test the number of measurements."""
  with db.Session() as s:
    q = s.query(me_db.Measurement) \
      .filter(me_db.Measurement.source == 'LifeCycle')

    # Test dataset has 25 entries - the first of which spans three days.
    assert q.count() == 28


def test_series_measurements_count(db: me_db.Database):
  """Test the number of measurements in Series."""
  with db.Session() as s:
    q = s.query(me_db.Measurement) \
      .filter(me_db.Measurement.source == 'LifeCycle') \
      .filter(me_db.Measurement.series == 'AlphaTime')

    # Test dataset has 25 entries - the first of which spans three days.
    assert q.count() == 28


def test_group_measurements_counts(db: me_db.Database):
  """Test the number of measurements in Series."""
  with db.Session() as s:
    q = s.query(me_db.Measurement) \
      .filter(me_db.Measurement.source == 'LifeCycle') \
      .filter(me_db.Measurement.series == 'AlphaTime') \
      .filter(me_db.Measurement.group == 'default')

    assert q.count() == 19

  with db.Session() as s:
    q = s.query(me_db.Measurement) \
      .filter(me_db.Measurement.source == 'LifeCycle') \
      .filter(me_db.Measurement.series == 'AlphaTime') \
      .filter(me_db.Measurement.group == 'Alpha')

    assert q.count() == 9


def test_date_seconds(db: me_db.Database):
  """Test the date field values."""
  # The start date (UTC) column of the test_inbox dataset begins at the given
  # datetime: 2017-01-20 01:55:01.
  start_date = datetime.datetime.strptime('2017-01-20 01:55:01',
                                          '%Y-%m-%d %H:%M:%S')
  with db.Session() as s:
    q = s.query(me_db.Measurement.date) \
      .filter(me_db.Measurement.source == 'LifeCycle') \
      .order_by(me_db.Measurement.date) \
      .limit(1)
    measurement_date, = q.one()
    assert measurement_date == start_date


if __name__ == '__main__':
  test.Main()
