"""Unit tests for //lib/labm8:prof."""
import os
import re
import sys

import pytest
from absl import app

from lib.labm8 import prof
from lib.labm8 import system

if system.is_python3():
  from io import StringIO
else:
  from StringIO import StringIO


@pytest.fixture
def profiling_env() -> None:
  """Create a session for an in-memory SQLite datastore.

  The database is will be empty.

  Returns:
    A Session instance.
  """
  try:
    os.environ['PROFILE'] = '1'
    yield None
  finally:
    os.environ['PROFILE'] = ''


def test_enable_disable():
  assert not prof.is_enabled()
  prof.disable()
  assert not prof.is_enabled()
  prof.enable()
  assert prof.is_enabled()
  prof.disable()
  assert not prof.is_enabled()


def test_named_timer(profiling_env):
  buf = StringIO()

  prof.start("foo")
  prof.stop("foo", file=buf)

  out = buf.getvalue()
  assert " foo " == re.search(" foo ", out).group(0)


def test_named_timer(profiling_env):
  buf = StringIO()

  prof.start("foo")
  prof.start("bar")
  prof.stop("bar", file=buf)

  out = buf.getvalue()
  assert not re.search(" foo ", out)
  assert " bar " == re.search(" bar ", out).group(0)

  prof.stop("foo", file=buf)

  out = buf.getvalue()
  assert " foo " == re.search(" foo ", out).group(0)
  assert " bar " == re.search(" bar ", out).group(0)


def test_stop_twice_error(profiling_env):
  prof.start("foo")
  prof.stop("foo")
  with pytest.raises(KeyError):
    prof.stop("foo")


def test_stop_bad_name_error(profiling_env):
  with pytest.raises(KeyError):
    prof.stop("not a timer")


def test_profile(profiling_env):
  def test_fn(x, y):
    return x + y

  assert prof.profile(test_fn, 1, 2) == 3


def test_timers(profiling_env):
  x = len(list(prof.timers()))
  prof.start("new timer")
  assert len(list(prof.timers())) == x + 1
  prof.stop("new timer")
  assert len(list(prof.timers())) == x


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
