from experimental.util.dotfiles.implementation import task


class shell(task.Task):
  def __init__(
    self, name=None, genfiles=None, deps=None, cmd=None, tmpfiles=None
  ):
    super(shell, self).__init__(name, genfiles, deps)
    self.cmd = task.AssertSet(cmd, "cmd")
    self.tmpfiles = tmpfiles or []
