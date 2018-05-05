"""Time utilities.
"""
import datetime

DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def strfmt(datetime, format=DEFAULT_DATETIME_FORMAT):
  """
  Format date to string.
  """
  return datetime.strftime(format)


def now():
  """
  Get the current datetime.
  """
  return datetime.datetime.now()


def nowstr(format=DEFAULT_DATETIME_FORMAT):
  """
  Convenience wrapper to get the current time as a string.

  Equivalent to invoking strfmt(now()).
  """
  return strfmt(now(), format=format)
