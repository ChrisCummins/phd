"""This module defines a utility function for cloning a git repo."""
import pathlib
import subprocess

from labm8.py import app

FLAGS = app.FLAGS


class RepoCloneFailed(OSError):
  """Error raised if repo fails to clone."""

  pass


def GitClone(
  clone_url: str,
  destination: pathlib.Path,
  shallow: bool = False,
  recursive: bool = False,
  timeout: int = 3600,
) -> pathlib.Path:
  """Clone a repository from Github.

  Args:
    clone_url: The URL of the repo to clone.
    destination: The output path. If this is already a non-empty directory, this
      will fail.
    shallow: Perform a shallow clone if set.
    recursive: Clone submodules.
    timeout: The maximum number of seconds to run a clone for before failing.

  Returns:
    The destination argument.

  Raises:
    RepoCloneFailed: On error.
  """
  cmd = [
    "timeout",
    "-s9",
    str(timeout),
    "git",
    "clone",
    clone_url,
    str(destination),
  ]
  if shallow:
    cmd += ["--depth", "1"]
  if recursive:
    cmd.append("--recursive")

  try:
    subprocess.check_call(cmd)
  except subprocess.CalledProcessError:
    raise RepoCloneFailed(f"Failed to clone repository: {clone_url}")
  if not (destination / ".git").is_dir():
    raise RepoCloneFailed(
      f"Cloned repo `{clone_url}` but `{destination}/.git` not found"
    )

  return destination
