"""Unit tests for workspace.py."""
import os
import pathlib
from tempfile import TemporaryDirectory

from labm8.py import app
from labm8.py import test
from util.photolib import workspace

FLAGS = app.FLAGS

MODULE_UNDER_TEST = "util.photolib.workspace"


def test_find_workspace_rootpath():
  """workspace.find_workspace_rootpath()"""
  assert not workspace.find_workspace_rootpath("")
  assert not workspace.find_workspace_rootpath("/not/a/real/path")

  with TemporaryDirectory() as tmpdir:
    os.mkdir(os.path.join(tmpdir, "photos"))
    os.mkdir(os.path.join(tmpdir, "photos", "2018"))
    os.mkdir(os.path.join(tmpdir, "photos", "2018", "2018-01"))
    os.mkdir(os.path.join(tmpdir, "third_party"))
    os.mkdir(os.path.join(tmpdir, "lightroom"))

    assert workspace.find_workspace_rootpath(tmpdir) == tmpdir

    # It can find the workspace in subdirectories.
    assert (
      workspace.find_workspace_rootpath(os.path.join(tmpdir, "photos"))
      == tmpdir
    )
    assert (
      workspace.find_workspace_rootpath(os.path.join(tmpdir, "third_party"))
      == tmpdir
    )
    assert (
      workspace.find_workspace_rootpath(os.path.join(tmpdir, "lightroom"))
      == tmpdir
    )
    assert (
      workspace.find_workspace_rootpath(
        os.path.join(tmpdir, "photos", "2018", "2018-01")
      )
      == tmpdir
    )

    # It can find the workspace even if the subdir is non-existent.
    assert (
      workspace.find_workspace_rootpath(os.path.join(tmpdir, "nondir"))
      == tmpdir
    )
    assert (
      workspace.find_workspace_rootpath(
        os.path.join(tmpdir, "nondir", "nondir")
      )
      == tmpdir
    )


def test_GetRelpath(tempdir: pathlib.Path):
  w = workspace.Workspace.Create(tempdir)
  assert w.GetRelpath(f"{tempdir}/b/c") == "//c"
  assert w.GetRelpath(f"{tempdir}/b/c/d") == "//c/d"

  # This is a strange test.
  assert w.GetRelpath(tempdir) == "/"


if __name__ == "__main__":
  test.Main()
