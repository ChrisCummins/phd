from experimental.util.dotfiles.implementation import task


class task_group(task.Task):
  """An abstract task used for grouping dependencies."""

  def __init__(self, name=None, deps=None):
    super(task_group, self).__init__(name, [], deps)
