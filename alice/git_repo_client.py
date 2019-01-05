"""Interface for manipulating the phd git repo."""
import pathlib

from absl import flags


FLAGS = flags.FLAGS


class GitRepoClient(object):

  def __init__(self, repo_root: pathlib.Path):
    del repo_root

  def CheckoutCommit(self, commit_hash: str) -> None:
    raise NotImplementedError

  @property
  def working_tree_is_clean(self):
    raise NotImplementedError

  @property
  def head_commit_hash(self) -> str:
    raise NotImplementedError
