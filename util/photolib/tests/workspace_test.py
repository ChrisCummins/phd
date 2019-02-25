"""Unit tests for workspace.py."""
import os
from tempfile import TemporaryDirectory

from absl import flags

from labm8 import test
from util.photolib import workspace

FLAGS = flags.FLAGS


def test_find_workspace_rootpath():
  """workspace.find_workspace_rootpath()"""
  assert not workspace.find_workspace_rootpath("")
  assert not workspace.find_workspace_rootpath("/not/a/real/path")

  with TemporaryDirectory() as tmpdir:
    os.mkdir(os.path.join(tmpdir, "photos"))
    os.mkdir(os.path.join(tmpdir, "photos", "2018"))
    os.mkdir(os.path.join(tmpdir, "photos", "2018", "2018-01"))
    os.mkdir(os.path.join(tmpdir, "gallery"))
    os.mkdir(os.path.join(tmpdir, "lightroom"))

    assert workspace.find_workspace_rootpath(tmpdir) == tmpdir

    # It can find the workspace in subdirectories.
    assert workspace.find_workspace_rootpath(os.path.join(tmpdir,
                                                          "photos")) == tmpdir
    assert workspace.find_workspace_rootpath(os.path.join(tmpdir,
                                                          "gallery")) == tmpdir
    assert workspace.find_workspace_rootpath(os.path.join(
        tmpdir, "lightroom")) == tmpdir
    assert workspace.find_workspace_rootpath(
        os.path.join(tmpdir, "photos", "2018", "2018-01")) == tmpdir

    # It can find the workspace even if the subdir is non-existent.
    assert workspace.find_workspace_rootpath(os.path.join(tmpdir,
                                                          "nondir")) == tmpdir
    assert workspace.find_workspace_rootpath(
        os.path.join(tmpdir, "nondir", "nondir")) == tmpdir


def test_get_workspace_relpath():
  """workspace.get_workspace_relpath()"""
  assert workspace.get_workspace_relpath("/a/b", "/a/b/c") == "//c"
  assert workspace.get_workspace_relpath("/a/b", "/a/b/c/d") == "//c/d"

  # This is a strange test.
  assert workspace.get_workspace_relpath("/a/b", "/a") == "/"


if __name__ == "__main__":
  test.Main()
