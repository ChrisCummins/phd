from experimental.util.dotfiles.implementation import host
from experimental.util.dotfiles.implementation import task


class symlink(task.Task):
  def __init__(self, name=None, deps=None, src=None, dst=None):
    super(symlink, self).__init__(name, [dst], deps)
    self.src = task.AssertSet(src, "src")
    self.dst = task.AssertSet(dst, "dst")

  def __call__(self, ctx):
    # _AssertIsFile(self.src)
    host.MakeSymlink(self.src, self.dst)
