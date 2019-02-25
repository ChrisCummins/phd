"""Unit tests for linters.py."""

import pytest
from absl import flags

from labm8 import test
from util.photolib import linters

FLAGS = flags.FLAGS


def test_error():
  # error() raises an assertion if the category is not recognized.
  with pytest.raises(AssertionError):
    linters.Error("//photos", "not/a/real/category", "msg")


def test_PhotolibFilename():
  linter = linters.PhotolibFilename()

  def good_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(f"/photos/{filename}.dng", f"//photos/{filename}.jpg",
           f"{filename}.jpg")
    assert linters.ERROR_COUNTS.get("file/name", 0) == n

  def bad_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(f"/photos/{filename}.dng", f"//photos/{filename}.jpg",
           f"{filename}.jpg")
    assert linters.ERROR_COUNTS.get("file/name", 0) == n + 1

  bad_name("foo")

  # ISO format.

  good_name("19700101T000101")
  good_name("20050101T000101")

  # Out-of-range years.
  bad_name("00000101T000101")
  bad_name("05000101T000101")
  bad_name("23000101T000101")

  # Out-of-range month.
  bad_name("19703001T000101")
  bad_name("19701501T000101")
  bad_name("19700001T000101")

  # Out-of-range day.
  bad_name("19700150T000101")
  bad_name("19700135T000101")
  bad_name("19700100T000101")

  # Sequence.
  good_name("19700101T000101-2")
  good_name("19700101T000101-1000")

  # Modifiers.
  bad_name("19700101T000101-foo")
  good_name("19700101T000101-Pano")
  good_name("19700101T000101-HDR")
  good_name("19700101T000101-Edit")
  good_name("19700101T000101-1-Edit")

  # Scan format.

  good_name("700101A-01")
  good_name("000101C-30")

  # Out-of-range month.
  bad_name("701301A-01")
  bad_name("700001A-01")

  # Out-of-range day.
  bad_name("701300A-01")
  bad_name("701340A-01")
  bad_name("701333A-01")

  # Sequence.
  good_name("000101A-30-1")
  good_name("000101A-30-1000")

  # Modifiers.
  bad_name("000101A-30-foo")
  good_name("000101A-30-Pano")
  good_name("000101A-30-HDR")
  good_name("000101A-30-Edit")
  good_name("000101A-30-1-Edit")


def test_GalleryFilename():
  """Checks that file name matches one of expected formats."""
  linter = linters.GalleryFilename()

  def good_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(f"/photos/{filename}.dng", f"//photos/{filename}.jpg",
           f"{filename}.jpg")
    assert linters.ERROR_COUNTS.get("file/name", 0) == n

  def bad_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(f"/photos/{filename}.dng", f"//photos/{filename}.jpg",
           f"{filename}.jpg")
    assert linters.ERROR_COUNTS.get("file/name", 0) == n + 1

  good_name("photos-1")

  # Contains whitespace.
  bad_name("photos 1")

  # Out of sequence (missing photos-1).
  bad_name("photos-2")


if __name__ == "__main__":
  test.Main()
