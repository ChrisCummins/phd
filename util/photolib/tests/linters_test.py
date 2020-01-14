"""Unit tests for linters.py."""
import pathlib

from labm8.py import app
from labm8.py import test
from util.photolib import contentfiles
from util.photolib import linters
from util.photolib import workspace
from util.photolib import xmp_cache

FLAGS = app.FLAGS

MODULE_UNDER_TEST = "util.photolib.linters"


@test.Fixture(scope="function")
def empty_workspace(tempdir: pathlib.Path):
  (tempdir / "workspace").mkdir()
  (tempdir / "workspace" / "WORKSPACE").touch()
  yield workspace.Workspace(str(tempdir / "workspace"))


def test_error():
  # error() raises an assertion if the category is not recognized.
  with test.Raises(AssertionError):
    linters.Error("//photos", "not/a/real/category", "msg")


def _MakePhoto(workspace_, filename):
  cf_path = workspace_.workspace_root / "photos" / f"{filename}.dng"
  return contentfiles.Contentfile(
    cf_path,
    workspace_.GetRelpath(str(cf_path)),
    filename,
    xmp_cache.XmpCache(workspace_),
  )


def test_PhotolibFilename(empty_workspace: workspace.Workspace):
  linter = linters.PhotolibFilename(empty_workspace)

  def good_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(_MakePhoto(empty_workspace, filename))
    assert linters.ERROR_COUNTS.get("file/name", 0) == n

  def bad_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(_MakePhoto(empty_workspace, filename))
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


def test_ThirdPartyFilename(empty_workspace: workspace.Workspace):
  """Checks that file name matches one of expected formats."""
  linter = linters.ThirdPartyFilename(empty_workspace)

  def good_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(_MakePhoto(empty_workspace, filename))
    assert linters.ERROR_COUNTS.get("file/name", 0) == n

  def bad_name(filename):
    n = linters.ERROR_COUNTS.get("file/name", 0)
    linter(_MakePhoto(empty_workspace, filename))
    assert linters.ERROR_COUNTS.get("file/name", 0) == n + 1

  good_name("photos-1")

  # Contains whitespace.
  bad_name("photos 1")

  # Out of sequence (missing photos-1).
  bad_name("photos-2")


if __name__ == "__main__":
  test.Main()
