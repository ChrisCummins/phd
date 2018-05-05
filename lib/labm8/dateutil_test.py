"""Unit tests for //lib/labm8:dateutil."""
import sys

import pytest
from absl import app

from lib.labm8 import dateutil


def test_GetUtcMillisecondsNow_millisecond_precision():
  # Test that milliseconds datetimes have no microseconds.
  now = dateutil.GetUtcMillisecondsNow()
  assert not now.microsecond % 1000


def test_MillisecondsTimestamp_invalid_argument():
  with pytest.raises(TypeError):
    dateutil.MillisecondsTimestamp('not a date')


def test_DatetimeFromMillisecondsTimestamp_invalid_argument():
  with pytest.raises(TypeError):
    dateutil.DatetimeFromMillisecondsTimestamp('not a timestamp')


def test_DatetimeFromMillisecondsTimestamp_negative_int():
  with pytest.raises(ValueError):
    dateutil.DatetimeFromMillisecondsTimestamp(-1)


def test_timestamp_datetime_equivalence():
  date_in = dateutil.GetUtcMillisecondsNow()
  timestamp = dateutil.MillisecondsTimestamp(date_in)
  date_out = dateutil.DatetimeFromMillisecondsTimestamp(timestamp)
  assert date_in == date_out


def test_default_timestamp_datetime_equivalence():
  now = dateutil.GetUtcMillisecondsNow()
  timestamp = dateutil.MillisecondsTimestamp()
  date_out = dateutil.DatetimeFromMillisecondsTimestamp(timestamp)
  assert now.date() == date_out.date()


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
