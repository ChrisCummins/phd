"""Unit tests for //TODO:alice/repo_state_test."""
import collections
import pathlib
import subprocess

import pytest

from alice import git_repo
from config import getconfig
from labm8 import pbutil
from labm8 import test


def _Git(gitdir: pathlib.Path, *args):
  assert gitdir.is_dir()
  subprocess.check_call(['git', '-C', gitdir] + list(args))


MockRepo = collections.namedtuple('MockRepo', ['path', 'remote'])


@pytest.fixture(scope='function')
def mock_repo(tempdir: pathlib.Path, tempdir2: pathlib.Path) -> MockRepo:
  """Test fixture that returns a mock repo with a remote."""
  remote_root = tempdir2 / 'remote'
  remote_root.mkdir()
  _Git(remote_root, 'init', '--bare')

  repo_root = tempdir / 'repo'
  repo_root.mkdir()
  _Git(repo_root, 'init')

  config = getconfig.GetGlobalConfig()
  pbutil.ToFile(config, repo_root / 'config.pbtxt')

  with open(repo_root / '.gitignore', 'w') as f:
    f.write('config.pbtxt\n')

  _Git(repo_root, 'add', '.gitignore')
  _Git(repo_root, 'config', 'user.email', 'test@example.com')
  _Git(repo_root, 'config', 'user.name', 'Test Man')

  _Git(repo_root, 'commit', '-m', 'Add git ignore')
  _Git(repo_root, 'remote', 'add', 'origin', str(remote_root))
  _Git(repo_root, 'push', '-u', 'origin', 'master')

  yield MockRepo(path=repo_root, remote=remote_root)


def test_PhdRepo_ToRepoState_clean(mock_repo: MockRepo):
  """Test values of state."""
  state = git_repo.PhdRepo(mock_repo.path).ToRepoState()
  assert state.remote_url == str(mock_repo.remote)
  assert state.tracking_branch == 'origin/master'


def test_PhdRepo_ToRepoState_FromRepoState_unchanged(mock_repo: MockRepo):
  """Test that state is unchanged after restoring from it."""
  repo = git_repo.PhdRepo(mock_repo.path)
  state0 = repo.ToRepoState()
  repo.FromRepoState(state0)
  state1 = repo.ToRepoState()
  assert state0 == state1


if __name__ == '__main__':
  test.Main()
