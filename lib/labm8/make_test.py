"""Unit tests for //lib/labm8:labmath."""
import sys

import pytest
from absl import app

from lib.labm8 import fs
from lib.labm8 import make


# make()
def test_make():
  ret, out, err = make.make(dir="lib/labm8/test/data/makeproj")
  assert 0 == ret
  assert True == out is not None
  assert True == fs.isfile("lib/labm8/test/data/makeproj/foo")
  assert True == fs.isfile("lib/labm8/test/data/makeproj/foo.o")


def test_make_bad_target():
  with pytest.raises(make.NoTargetError):
    make.make(target="bad-target", dir="lib/labm8/test/data/makeproj")


def test_make_bad_target():
  with pytest.raises(make.NoMakefileError):
    make.make(dir="/bad/path")
  with pytest.raises(make.NoMakefileError):
    make.make(target="foo", dir="lib/labm8/test/data")


def test_make_fail():
  with pytest.raises(make.MakeError):
    make.make(target="fail", dir="lib/labm8/test/data/makeproj")


# clean()
def test_make_clean():
  fs.cd("lib/labm8/test/data/makeproj")
  make.make()
  assert fs.isfile("foo")
  assert fs.isfile("foo.o")
  make.clean()
  assert not fs.isfile("foo")
  assert not fs.isfile("foo.o")
  fs.cdpop()


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
