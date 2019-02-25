"""Utilities for working with workspaces."""
import os
import typing

from util.photolib import common


def find_workspace_rootpath(start_path: str) -> typing.Optional[str]:
  """
  Starting at the given path, ascend up the directory tree until the
  workspace root is found.

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


def get_workspace_relpath(workspace_root: str, abspath: str) -> str:
  """
  Convert an absolute path into a workspace-relative path.

  Args:
    workspace_root: The absolute path to the workspace root.
    abspath: An absolute path.

  Returns:
    A workspace path, i.e. one in which the root of the workspace has been
    replaced by '//'
  """
  return '/' + abspath[len(workspace_root):]
