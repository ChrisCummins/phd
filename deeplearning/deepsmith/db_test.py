"""Unit tests for :db."""
import sys

import pytest
from absl import app

from deeplearning.deepsmith import db
from deeplearning.deepsmith import toolchain


def test_StringTable_GetOrAdd_StringTooLongError(session):
  toolchain.Toolchain.GetOrAdd(session, 'a' * toolchain.Toolchain.maxlen)
  with pytest.raises(db.StringTooLongError):
    toolchain.Toolchain.GetOrAdd(
        session, 'a' * (toolchain.Toolchain.maxlen + 1))


def test_StringTable_TruncatedString(session):
  t = toolchain.Toolchain.GetOrAdd(session, 'a' * 80)
  assert t.TruncatedString() == 'a' * 80
  assert len(t.TruncatedString(n=70)) == 70
  assert t.TruncatedString(n=70) == 'a' * 67 + '...'


def test_StringTable_TruncatedString_uninitialized():
  t = toolchain.Toolchain()
  assert len(t.TruncatedString()) == 0


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
  app.run(main)
