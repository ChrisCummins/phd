"""Access to the current repo build."""

import datetime
import pathlib
import re
from typing import Optional

import git

from config import getconfig
from labm8 import app

FLAGS = app.FLAGS


class BuildInfo(object):
  """A class which encapsulates state about the current repo build."""

  def __init__(self, repo: git.Repo):
    head = repo.head.object
    branch = repo.active_branch
    self.id: str = head.hexsha
    self.date: datetime.datetime = datetime.datetime.fromtimestamp(
        head.authored_date)
    self.author: str = f"{head.author.name} <{head.author.email}>"
    self.dirty: bool = repo.is_dirty()
    self.branch: str = branch.name

    tracking = branch.tracking_branch()
    if tracking:
      remote = tracking.remote_name
      self.remote: Optional[str] = remote
      self.commit_url: Optional[str] = _GetGitHubCommitUrl(
          repo.remote(remote), self.id)
    else:
      self.remote: Optional[str] = None
      self.commit_url: Optional[str] = None

  @property
  def short_hash(self) -> str:
    return self.id[:7]


def GetBuildInfo() -> BuildInfo:
  """Return the current repo build state."""
  return BuildInfo(GetGitRepo())


def GetGitRepo(
    config_path: pathlib.Path = getconfig.GLOBAL_CONFIG_PATH) -> git.Repo:
  """Get the git repo for this project."""
  config = getconfig.GetGlobalConfig(path=config_path)
  assert config.paths.repo_root
  return git.Repo(path=config.paths.repo_root)


def _GetGitHubCommitUrl(remote: git.Remote, hexsha: str):
  """Calculate the GitHub URL for a commit."""
  m = re.match(f'git@github\.com:([^/]+)/(.+)\.git', remote.url)
  if not m:
    return None
  return f'https://github.com/{m.group(1)}/{m.group(2)}/commit/{hexsha}'
