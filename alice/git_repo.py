"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import pathlib

import git
from absl import flags

from alice import alice_pb2

FLAGS = flags.FLAGS


class PhdRepo(object):

  def __init__(self, path: pathlib.Path):
    self._path = path
    self._repo = git.Repo(path=path)

  @property
  def repo(self) -> git.Repo:
    return self._repo

  @property
  def path(self) -> pathlib.Path:
    return self._path

  def ToRepoState(self) -> alice_pb2.RepoState:
    assert not self.repo.head.is_detached

    tracking_branch = self.repo.active_branch.tracking_branch()
    assert tracking_branch

    remote_urls = list(self.repo.remote(tracking_branch.remote_name).urls)
    assert len(remote_urls) >= 1

    return alice_pb2.RepoState(
        remote_url=remote_urls[0],
        tracking_branch=tracking_branch.name,
        head_id=self.repo.head.object.hexsha,
        # TODO(cec): Assemble and return git diff.
    )

  def FromRepoState(self, repo_state: alice_pb2.RepoState) -> None:
    remote = self.repo.create_remote('alice_tmp_remote', repo_state.remote_url)
    assert remote.exists()
    remote.fetch()
    commit = repo_state.head_id
    self.repo.head.set_commit(commit)
