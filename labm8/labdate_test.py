"""Unit tests for //labm8:labdate."""

import pytest
from absl import flags

from labm8 import labdate
from labm8 import test

FLAGS = flags.FLAGS


def test_GetUtcMillisecondsNow_millisecond_precision():
  # Test that milliseconds datetimes have no microseconds.
  now = labdate.GetUtcMillisecondsNow()
  assert not now.microsecond % 1000


def test_MillisecondsTimestamp_invalid_argument():
  with pytest.raises(TypeError):
    labdate.MillisecondsTimestamp('not a date')


def test_DatetimeFromMillisecondsTimestamp_default_argument():
  """Current time is used if no timestamp provided."""
  assert labdate.DatetimeFromMillisecondsTimestamp()


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


if __name__ == '__main__':
  test.Main()
