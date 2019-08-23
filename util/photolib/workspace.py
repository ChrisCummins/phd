"""Utilities for working with workspaces."""
import os

import pathlib
import typing

from labm8 import app
from util.photolib import common

FLAGS = app.FLAGS


def find_workspace_rootpath(start_path: str) -> typing.Optional[str]:
  """
  Starting at the given path, ascend up the directory tree until the
  workspace root is found.

  TODO: Delete this, replace with Workspace.FindWorkspace().

  Args:
      start_path: The starting path.

  Returns:
      The path of the workspace root, or None if not found.
  """
  if all(
      os.path.isdir(os.path.join(start_path, tld))
      for tld in common.TOP_LEVEL_DIRS):
    return os.path.abspath(start_path)

  if os.path.ismount(start_path):
    return None
  else:
    up_one_dir = os.path.abspath(os.path.join(start_path, ".."))
    return find_workspace_rootpath(up_one_dir)


class Workspace(object):
  """A photolib workspace."""

  def __init__(self, root_path: str):
    self.workspace_root = pathlib.Path(root_path)
    if not (self.workspace_root / 'WORKSPACE').is_file():
      raise FileNotFoundError(
          f"File not found: '{self.workspace_root / 'WORKSPACE'}'")

  def GetRelpath(self, abspath: str) -> str:
    """Convert an absolute path into a workspace-relative path.

    Args:
      abspath: An absolute path.

    Returns:
      A workspace path, i.e. one in which the root of the workspace has been
      replaced by '//'
    """
    return '/' + abspath[len(str(self.workspace_root)):]

  @classmethod
  def Create(cls, root_path: pathlib.Path):
    if not root_path.is_dir():
      raise FileNotFoundError(f"Workspace root not found: `{root_path}`")

    (root_path / 'WORKSPACE').touch()
    (root_path / '.photolib').mkdir()
    (root_path / 'photos').mkdir()
    (root_path / 'third_party').mkdir()
    (root_path / 'lightroom').mkdir()

  @classmethod
  def FindWorkspace(cls, path: pathlib.Path) -> 'Workspace':
    """Look for and return a workspace at or above the current path.

    Args:
      path: The path to a file or directory to start the search from.

    Raises:
      FileNotFoundError: If no workspace is found by the time the nearest mount
        point is reached.
    """
    if (path / 'WORKSPACE').is_file():
      app.Log(2, 'Found workspace: `%s`', path)
      return Workspace(path)
    elif path.is_mount():
      raise FileNotFoundError("Workspace not found")
    else:
      return cls.FindWorkspace(path.parent)
