"""Unit tests for linters.py."""
import pytest
import sys

from absl import app
from absl import flags

from photolib import linters


def test_error():
    # error() raises an assertion if the category is not recognized.
    with pytest.raises(AssertionError):
        linters.error("//photos", "not/a/real/category", "msg")


def test_PhotolibFilename():
    linter = linters.PhotolibFilename()
    assert 1 <= linter.cost <= 100

    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter("/photos/foo.jpg", "//photos/foo.jpg", "foo.jpg")
    assert linters.ERROR_COUNTS.get("file/name", 0) == n + 1

    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter("/photos/foo.jpg", "//photos/foo.jpg", "foo.jpg")
    assert linters.ERROR_COUNTS.get("file/name", 0) == n + 1


def main(argv):  # pylint: disable=missing-docstring
    del argv
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    app.run(main)
