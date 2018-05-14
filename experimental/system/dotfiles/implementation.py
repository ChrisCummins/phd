"""This file implements the declarative functions for dotfiles."""
from __future__ import print_function


class TaskArgumentError(Exception):

  def __init__(self, argument, message):
    self.argument = argument
    self.message = message

  def __repr__(self):
    return ' '.join([self.argument, self.message])


class MissingRequiredArgument(TaskArgumentError):
  def __init__(self, argument):
    super(MissingRequiredArgument, self).__init__(argument, 'not set')


def _AssertSet(name, value):
  if value is None:
    raise MissingRequiredArgument(name)
  return value


class _task(object):

  def __init__(self, name=None, deps=None):
    print(self, "__init__")
    self.name = _AssertSet(name, 'name')
    self.deps = _AssertSet(deps, 'deps')

  def __call__(self):
    print(self, "__call__")
    pass


class task_group(_task):
  """An abstract task used for grouping dependencies."""

  def __init__(self, name=None, deps=None):
    super(task_group, self).__init__(name, deps)


class brew_package(_task):

  def __init__(self, name=None, deps=None, package=None):
    super(brew_package, self).__init__(name, deps)
    self.package = _AssertSet(package)

  def __call__(self):
    brew = Homebrew()
    brew.install()
    brew.install_package(self.package)


class symlink(_task):

  def __init__(self, name=None, deps=None, src=None, dst=None):
    super(symlink, self).__init__(name, deps)
    self.src = _AssertSet(src, 'src')
    self.dst = _AssertSet(dst, 'dst')

  def __call__(self):
    _AssertIsFile(self.src)
    make_symlink(self.src, self.dst)


class github_repo(_task):

  def __init__(self, name=None, deps=None, ssh_remote=None, https_remote=None,
               dst=None, head=None):
    super(github_repo, self).__init__(name, deps)
    self.ssh_remote = ssh_remote
    self.https_remote = https_remote
    self.dst = _AssertSet(dst, 'dst')
    self.head = head


class shell(_task):

  def __init__(self, name=None, deps=None, src=None):
    super(shell, self).__init__(name, deps)
    self.src = _AssertSet(src)
