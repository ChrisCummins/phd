"""Unite tests for //tools/git:git_remote."""
import git

from labm8.py import test
from tools.git import git_remote

pytest_plugins = ["tools.git.test.fixtures"]

FLAGS = test.FLAGS


def test_GitRemote_remote_is_available(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  temporary_remote = git_remote.GitRemote(
    empty_repo, repo_with_history.working_tree_dir
  )
  with temporary_remote as temporary_remote_name:
    assert empty_repo.remote(temporary_remote_name)


def test_GitRemote_remote_fetch(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  temporary_remote = git_remote.GitRemote(
    empty_repo, repo_with_history.working_tree_dir
  )
  with temporary_remote as temporary_remote_name:
    empty_repo.remote(temporary_remote_name).fetch()


def test_GitRemote_remote_is_removed_after_scope(
  repo_with_history: git.Repo, empty_repo: git.Repo
):
  temporary_remote = git_remote.GitRemote(
    empty_repo, repo_with_history.working_tree_dir
  )
  with temporary_remote as temporary_remote_name:
    pass
  assert temporary_remote_name not in empty_repo.remotes


if __name__ == "__main__":
  test.Main()
