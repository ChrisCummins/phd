"""Unit tests for //labm8:decorators.py."""

import time

import pytest

from labm8 import app
from labm8 import decorators
from labm8 import test

FLAGS = app.FLAGS


class DummyClass(object):

  def __init__(self):
    self.memoized_property_run_count = 0

  @decorators.memoized_property
  def memoized_property(self):
    self.memoized_property_run_count += 1
    # In "real world" usage, this would be an expensive computation who's result
    # we would like to memoize.
    return 5


def test_memoized_property_value():
  """Test that memoized property returns expected value."""
  c = DummyClass()
  assert c.memoized_property == 5


def test_memoized_property_run_count():
  """Test that repeated access to property returns memoized value."""
  c = DummyClass()
  _ = c.memoized_property
  _ = c.memoized_property
  _ = c.memoized_property
  assert c.memoized_property_run_count == 1


def test_timeout_without_exception_timeout_not_raised():
  """Test that decorated function runs."""

  @decorators.timeout_without_exception(seconds=1)
  def Func() -> int:
    """Function under test."""
    return 5

  assert Func() == 5


def test_timeout_without_exception_timeout_raised():
  """Test that decorated function doesn't raise exception."""

  @decorators.timeout_without_exception(seconds=1)
  def Func() -> int:
    """Function under test."""
    time.sleep(10)
    return 5

  assert not Func()


def test_timeout_timeout_not_raised():
  """Test that decorated function doesn't raise exception."""

  @decorators.timeout(seconds=1)
  def Func() -> int:
    """Function under test."""
    return 5

  assert Func() == 5


def test_timeout_timeout_raised():
  """Test that decorated function doesn't raise exception."""

  @decorators.timeout(seconds=1)
  def Func() -> int:
    """Function under test."""
    time.sleep(10)
    return 5

  with pytest.raises(TimeoutError):
    Func()


if __name__ == '__main__':
  test.Main()
