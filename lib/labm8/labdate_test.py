"""Unit tests for //lib/labm8:labdate."""
import sys

import pytest
from absl import app

from phd.lib.labm8 import labdate


def test_GetUtcMillisecondsNow_millisecond_precision():
  # Test that milliseconds datetimes have no microseconds.
  now = labdate.GetUtcMillisecondsNow()
  assert not now.microsecond % 1000


def test_MillisecondsTimestamp_invalid_argument():
  with pytest.raises(TypeError):
    labdate.MillisecondsTimestamp('not a date')


def test_DatetimeFromMillisecondsTimestamp_invalid_argument():
  with pytest.raises(TypeError):
    labdate.DatetimeFromMillisecondsTimestamp('not a timestamp')


def test_DatetimeFromMillisecondsTimestamp_negative_int():
  with pytest.raises(ValueError):
    labdate.DatetimeFromMillisecondsTimestamp(-1)


def test_timestamp_datetime_equivalence():
  date_in = labdate.GetUtcMillisecondsNow()
  timestamp = labdate.MillisecondsTimestamp(date_in)
  date_out = labdate.DatetimeFromMillisecondsTimestamp(timestamp)
  assert date_in == date_out


def test_default_timestamp_datetime_equivalence():
  now = labdate.GetUtcMillisecondsNow()
  timestamp = labdate.MillisecondsTimestamp()
  date_out = labdate.DatetimeFromMillisecondsTimestamp(timestamp)
  assert now.date() == date_out.date()


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
