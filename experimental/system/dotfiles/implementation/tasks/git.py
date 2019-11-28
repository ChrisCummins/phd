import os

from experimental.system.dotfiles.implementation import task


class github_repo(task.Task):
  def __init__(
    self,
    name=None,
    deps=None,
    ssh_remote=None,
    https_remote=None,
    dst=None,
    head=None,
  ):
    super(github_repo, self).__init__(name, [os.path.join(dst, ".git")], deps)
    self.ssh_remote = ssh_remote
    self.https_remote = https_remote
    self.dst = task.AssertSet(dst, "dst")
    self.head = head
