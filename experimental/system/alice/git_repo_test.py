# Copyright (c) 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# alice is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alice.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //experimental/system/alice:git_repo."""
import pathlib
import subprocess
import typing

import pytest

import getconfig
from experimental.system.alice import git_repo
from labm8.py import pbutil
from labm8.py import test


def _Git(gitdir: pathlib.Path, *args):
  assert gitdir.is_dir()
  subprocess.check_call(["git", "-C", gitdir] + list(args))


class MockRepo(typing.NamedTuple):
  path: pathlib.Path
  remote: pathlib.Path


@test.Fixture(scope="function")
def mock_repo(tempdir: pathlib.Path, tempdir2: pathlib.Path) -> MockRepo:
  """Test fixture that returns a mock repo with a remote."""
  remote_root = tempdir2 / "remote"
  remote_root.mkdir()
  _Git(remote_root, "init", "--bare")

  repo_root = tempdir / "repo"
  repo_root.mkdir()
  _Git(repo_root, "init")

  config = getconfig.GetGlobalConfig()
  pbutil.ToFile(config, repo_root / "config.pbtxt")

  with open(repo_root / ".gitignore", "w") as f:
    f.write("config.pbtxt\n")

  _Git(repo_root, "add", ".gitignore")
  _Git(repo_root, "config", "user.email", "test@example.com")
  _Git(repo_root, "config", "user.name", "Test Man")

  _Git(repo_root, "commit", "-m", "Add git ignore")
  _Git(repo_root, "remote", "add", "origin", str(remote_root))
  _Git(repo_root, "push", "-u", "origin", "master")

  yield MockRepo(path=repo_root, remote=remote_root)


def test_PhdRepo_ToRepoState_clean(mock_repo: MockRepo):
  """Test values of state."""
  state = git_repo.PhdRepo(mock_repo.path).ToRepoState()
  assert state.remote_url == str(mock_repo.remote)
  assert state.tracking_branch == "origin/master"


def test_PhdRepo_ToRepoState_FromRepoState_unchanged(mock_repo: MockRepo):
  """Test that state is unchanged after restoring from it."""
  repo = git_repo.PhdRepo(mock_repo.path)
  state0 = repo.ToRepoState()
  repo.FromRepoState(state0)
  state1 = repo.ToRepoState()
  assert state0 == state1


if __name__ == "__main__":
  test.Main()
