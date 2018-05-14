"""This file implements the declarative functions for dotfiles."""
from __future__ import print_function

import sys
import time

import argparse
import collections
import logging
import os
import platform
import socket
import subprocess
from distutils import spawn

EXCLUDES = []
LINUX_DISTROS = ['debian', 'ubuntu']


class InvalidTaskError(Exception):
  pass


class DuplicateTaskName(InvalidTaskError):
  pass


class TaskArgumentError(InvalidTaskError):

  def __init__(self, argument, message):
    self.argument = argument
    self.message = message

  def __str__(self):
    return ' '.join([self.argument, self.message])


class MissingRequiredArgument(TaskArgumentError):
  def __init__(self, argument):
    super(MissingRequiredArgument, self).__init__(argument, 'not set')


class SchedulingError(Exception):
  pass


class CalledProcessError(EnvironmentError):
  pass


class Colors(object):
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'


def _AssertSet(value, name):
  if value is None:
    raise MissingRequiredArgument(name)
  return value


def GetPlatform():
  distro = platform.linux_distribution()
  if not distro[0]:
    return {
      "darwin": "osx",
    }.get(sys.platform, sys.platform)
  else:
    return {
      "debian": "ubuntu",
    }.get(distro[0].lower(), distro[0].lower())


def which(binary):
  return spawn.find_executable(binary)


def task_print(*msg, **kwargs):
  sep = kwargs.get("sep", " ")
  text = sep.join(msg)
  indented = "\n        > ".join(text.split("\n"))
  logging.info(Colors.GREEN + "        > " + indented + Colors.END)


def _log_shell(*args):
  if logging.getLogger().level <= logging.INFO:
    logging.debug(Colors.PURPLE + "    $ " + "".join(*args) + Colors.END)


def _log_shell_output(stdout):
  if logging.getLogger().level <= logging.INFO and len(stdout):
    indented = '    ' + '\n    '.join(stdout.rstrip().split('\n'))
    logging.debug(Colors.YELLOW + indented + Colors.END)


def _shell(*args):
  """ run a shell command and return its output. Raises CalledProcessError
      if fails """
  _log_shell(*args)
  p = subprocess.Popen(*args, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
  stdout, _ = p.communicate()

  stdout = stdout.rstrip()
  _log_shell_output(stdout)

  if p.returncode:
    cmd = " ".join(args)
    msg = ("""\
Command '{cmd}' failed with returncode {p.returncode} and output:
{stdout}""".format(cmd=cmd, p=p, stdout=stdout))
    raise CalledProcessError(msg)
  else:
    return stdout


def shell_ok(*args):
  """ run a shell command and return False if error """
  _log_shell(*args)
  try:
    subprocess.check_call(*args, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    _log_shell_output("-> 0")
    return True
  except subprocess.CalledProcessError as e:
    _log_shell_output("-> " + str(e.returncode))
    return False


def MakeSymlink(src, dst, sudo=False):
  src = os.path.expanduser(src)
  dst = os.path.expanduser(dst)

  if src.startswith("/"):
    src_abs = src
  else:
    src_abs = os.path.dirname(dst) + "/" + src

  # Symlink already exists
  use_sudo = "sudo -H " if sudo else ""
  if (shell_ok("{use_sudo}test -f '{dst}'".format(**vars())) or
      shell_ok("{use_sudo}test -d '{dst}'".format(**vars()))):
    linkdest = _shell("{use_sudo}readlink {dst}".format(**vars())).rstrip()
    if linkdest.startswith("/"):
      linkdest_abs = linkdest
    else:
      linkdest_abs = os.path.dirname(dst) + "/" + linkdest
    if linkdest_abs == src_abs:
      return

  if not (shell_ok("{use_sudo}test -f '{src_abs}'".format(**vars())) or
          shell_ok("{use_sudo}test -d '{src_abs}'".format(**vars()))):
    raise OSError("symlink source '{src}' does not exist".format(**vars()))
  # if shell_ok("{use_sudo}test -d '{dst}'".format(**vars())):
  #     raise OSError("symlink destination '{dst}' is a directory".format(**vars()))

  # Make a backup of existing file:
  if (shell_ok("{use_sudo}test -f '{dst}'".format(**vars())) or
      shell_ok("{use_sudo}test -d '{dst}'".format(**vars()))):
    _shell("{use_sudo}mv '{dst}' '{dst}'.backup".format(**vars()))

  # in case of broken symlink
  _shell("{use_sudo}rm -f '{dst}'".format(**vars()))

  # Create the symlink:
  task_print("Creating symlink {dst}".format(**vars()))
  _shell("{use_sudo}ln -s '{src}' '{dst}'".format(**vars()))


def get_task_method(task, method_name):
  """ resolve task method. First try and match <method>_<platform>(),
      then <method>() """
  fn = getattr(task, method_name + "_" + GetPlatform(), None)
  if fn is None and GetPlatform() in LINUX_DISTROS:
    fn = getattr(task, method_name + "_linux", None)
  if fn is None:
    fn = getattr(task, method_name, None)
  if fn is None:
    raise InvalidTaskError("failed to resolve {method_name} method of Task {task}".format(**vars()))
  return fn


class Homebrew(object):
  """ homebrew package manager """
  # Temporary files for caching list of installed packages and casks
  PKG_LIST = os.path.abspath(".brew-pkgs.txt")
  CASK_LIST = os.path.abspath(".brew-casks.txt")
  OUTDATED_PKG_LIST = os.path.abspath(".brew-pkgs-outdated.txt")
  OUTDATED_CASK_LIST = os.path.abspath(".brew-casks-outdated.txt")

  BREW_BINARY = {
    'osx': '/usr/local/bin/brew',
    'ubuntu': '/home/linuxbrew/.linuxbrew/bin/brew',
  }[GetPlatform()]

  __platforms__ = ['linux', 'osx']
  __deps__ = []
  __genfiles__ = [BREW_BINARY]
  __tmpfiles__ = [PKG_LIST, CASK_LIST, OUTDATED_PKG_LIST, OUTDATED_CASK_LIST]

  @classmethod
  def install(cls):
    if GetPlatform() == 'osx':
      cls.install_osx()
    elif GetPlatform() == 'ubuntu':
      cls.install_ubuntu()

  @staticmethod
  def install_osx():
    if not which('brew'):
      task_print("Installing Homebrew")
      url = 'https://raw.githubusercontent.com/Homebrew/install/master/install'
      _shell('yes | /usr/bin/ruby -e "$(curl -fsSL {url})"'.format(**vars()))
      _shell('brew doctor')

  @classmethod
  def install_ubuntu(cls):
    # Install build dependencies:
    Apt().install_package("build-essential")
    Apt().install_package("curl")
    Apt().install_package("file")
    Apt().install_package("git")
    Apt().install_package("python-setuptools")

    if not os.path.exists('/home/linuxbrew/.linuxbrew/bin/brew'):
      url = ("https://raw.githubusercontent.com/"
             "Linuxbrew/install/master/install.sh")
      _shell('yes | sh -c "$(curl -fsSL {url})"'.format(url=url))
      _shell('{brew} update'.format(brew=cls.BREW_BINARY))

  def package_is_installed(self, package):
    """ return True if package is installed """
    if not os.path.isfile(self.PKG_LIST):
      _shell("{self.BREW_BINARY} list > {self.PKG_LIST}".format(**vars()))

    return shell_ok("grep '^{package}$' <{self.PKG_LIST}".format(**vars()))

  def install_package(self, package):
    """ install a package using homebrew, return True if installed """
    if not self.package_is_installed(package):
      task_print("brew install " + package)
      _shell("{self.BREW_BINARY} install {package}".format(**vars()))
      return True

  def package_is_outdated(self, package):
    """ returns True if package is outdated """
    if not self.package_is_installed(package):
      raise InvalidTaskError("homebrew package '{package}' cannot be upgraded "
                             "as it is not installed".format(**vars()))

    if not os.path.isfile(self.OUTDATED_PKG_LIST):
      _shell("{self.BREW_BINARY} outdated | awk '{{print $1}}' >{self.OUTDATED_PKG_LIST}"
             .format(**vars()))

    package_stump = package.split('/')[-1]
    return shell_ok("grep '^{package_stump}$' <{self.OUTDATED_PKG_LIST}".format(**vars()))

  def upgrade_package(self, package):
    """ upgrade package, return True if upgraded """
    if self.package_is_outdated(package):
      task_print("brew upgrade {package}".format(**vars()))
      _shell("{self.BREW_BINARY} upgrade {package}".format(**vars()))
      return True

  def cask_is_installed(self, cask):
    """ return True if cask is installed """
    if not os.path.isfile(self.CASK_LIST):
      _shell("{self.BREW_BINARY} cask list > {self.CASK_LIST}".format(**vars()))

    cask_stump = cask.split('/')[-1]
    return shell_ok("grep '^{cask_stump}$' <{self.CASK_LIST}".format(**vars()))

  def install_cask(self, cask):
    """ install a homebrew cask, return True if installed """
    if not self.cask_is_installed(cask):
      task_print("brew cask install " + cask)
      _shell("{self.BREW_BINARY} cask install {cask}".format(**vars()))
      return True

  def cask_is_outdated(self, cask):
    """ returns True if cask is outdated """
    if not self.cask_is_installed(cask):
      raise InvalidTaskError(
        "homebrew cask '{package}' cannot be upgraded as it is not installed"
          .format(**vars()))

    if not os.path.isfile(self.OUTDATED_CASK_LIST):
      _shell("{self.BREW_BINARY} cask outdated ".format(**vars()) +
             "| awk '{{print $1}}' >{self.OUTDATED_CASK_LIST}".format(**vars()))

    cask_stump = cask.split('/')[-1]
    return shell_ok("grep '^{cask_stump}$' <{self.OUTDATED_CASK_LIST}".format(**vars()))

  def upgrade_cask(self, cask):
    """ upgrade a homebrew cask. does nothing if cask not installed """
    if self.cask_is_outdated(cask):
      task_print("brew cask upgrade {cask}".format(**vars()))
      _shell("{self.BREW_BINARY} cask upgrade {cask}".format(**vars()))
      return True

  def uninstall_cask(self, cask):
    """ remove a homebrew cask, return True if uninstalled """
    if self.cask_is_installed(cask):
      task_print("brew cask remove " + cask)
      _shell("{self.BREW_BINARY} cask remove " + cask)
      return True

  @staticmethod
  def _home():
    if GetPlatform() == 'osx':
      return '/usr/local'
    else:
      return '/home/linuxbrew/.linuxbrew'

  @classmethod
  def bin(cls, name):
    home = cls._home()
    return '{home}/bin/{name}'.format(home=home, name=name)

  @classmethod
  def lib(cls, name):
    home = cls._home()
    return '{home}/lib/{name}'.format(home=home, name=name)


class Apt(object):
  """ debian package manager """

  def install_package(self, package):
    """ install a package using apt-get, return True if installed """
    if not shell_ok("dpkg -s '{package}' &>/dev/null".format(package=package)):
      _shell("sudo apt-get install -y '{package}'".format(package=package))
      return True

  def update(self):
    """ update package information """
    _shell("sudo apt-get update")


TASKS = {}


class CallContext(object):
  pass


class _task(object):

  def __init__(self, name=None, genfiles=None, deps=None):
    print(self, "__init__")
    if name in TASKS:
      raise DuplicateTaskName(name)
    TASKS[name] = self
    self.name = _AssertSet(name, 'name')
    self.genfiles = genfiles or []
    self.deps = deps or []

  def SetUp(self, ctx):
    pass

  def __call__(self, ctx):
    pass

  def TearDown(self, ctx):
    pass


class task_group(_task):
  """An abstract task used for grouping dependencies."""

  def __init__(self, name=None, deps=None):
    super(task_group, self).__init__(name, [], deps)


class brew_package(_task):

  def __init__(self, name=None, genfiles=None, deps=None, package=None,
               force_link=None):
    super(brew_package, self).__init__(name, genfiles, deps)
    self.package = _AssertSet(package, 'package')
    self.force_link = force_link

  def __call__(self, ctx):
    package = self.package
    brew = Homebrew()
    brew.install()
    if brew.install_package(package):
      _shell("{brew.BREW_BINARY} link {package} --force".format(
        brew=brew, package=package))


class symlink(_task):

  def __init__(self, name=None, deps=None, src=None, dst=None):
    super(symlink, self).__init__(name, [dst], deps)
    self.src = _AssertSet(src, 'src')
    self.dst = _AssertSet(dst, 'dst')

  def __call__(self, ctx):
    # _AssertIsFile(self.src)
    MakeSymlink(self.src, self.dst)


class github_repo(_task):

  def __init__(self, name=None, deps=None, ssh_remote=None, https_remote=None,
               dst=None, head=None):
    super(github_repo, self).__init__(name, [os.path.join(dst, '.git')], deps)
    self.ssh_remote = ssh_remote
    self.https_remote = https_remote
    self.dst = _AssertSet(dst, 'dst')
    self.head = head


class shell(_task):

  def __init__(self, name=None, genfiles=None, deps=None, cmd=None):
    super(shell, self).__init__(name, genfiles, deps)
    self.cmd = _AssertSet(cmd, 'cmd')


def IsCompatible(a, b):
  """ return if platforms a and b are compatible """
  if b == "linux":
    return a in LINUX_DISTROS or a == "linux"
  else:
    return a == b


def IsRunnableTask(task):
  """ returns true if object is a task for the current platform """
  # Check that object is a class and inherits from 'Task':
  if not isinstance(task, _task):
    return False

  # Check that hostname is whitelisted (if whitelist is provided):
  hosts = getattr(task, "__hosts__", [])
  if hosts and socket.gethostname() not in hosts:
    msg = "skipping " + type(task).__name__ + " on host " + socket.gethostname()
    logging.debug(msg)
    return False

  # Check that task passes all req tests:
  reqs = getattr(task, "__reqs__", [])
  if reqs and not all(req() for req in reqs):
    msg = "skipping " + type(task).__name__ + ", failed req check"
    logging.debug(msg)
    return False

  return True


def ScheduleTask(task_name, schedule, all_tasks, depth=1):
  """ recursively schedule a task and its dependencies """
  # Sanity check for scheduling errors:
  if depth > 1000:
    raise SchedulingError("failed to resolve schedule for task '" +
                          task_name + "' after 1000 tries")
    sys.exit(1)

  # Instantiate the task class:
  task = all_tasks.get(task_name)
  if not task:
    raise SchedulingError("task '" + task_name + "' not found!")

  # If the task is not runnable, schedule nothing.
  if not IsRunnableTask(task):
    print('not runnable')
    return True

  # Check that all dependencies have already been scheduled:
  for dep_name in task.deps:
    if dep_name == task_name:
      raise SchedulingError("task '" + task_name + "' depends on itself")

    if dep_name not in schedule:
      # If any of the dependencies are not runnable, schedule nothing.
      if ScheduleTask(dep_name, schedule, all_tasks, depth + 1):
        return True

  # Schedule the task if necessary:
  if task_name not in schedule:
    schedule.append(task_name)


def GetTasksToRun(task_names):
  """ generate the list of task names to run """
  # Remove duplicate task names:
  task_names = set(task_names)

  # Determine the tasks which need scheduling:
  to_schedule = task_names if len(task_names) else TASKS.keys()

  # Build the schedule:
  to_schedule = collections.deque(sorted(to_schedule))
  schedule = []
  try:
    while len(to_schedule):
      task = to_schedule.popleft()
      ScheduleTask(task, schedule, TASKS)
  except SchedulingError as e:
    logging.critical("fatal: " + str(e))
    sys.exit(1)

  print("LEN sched", len(schedule))
  return collections.deque(schedule)


def brew_bin(name):
  return Homebrew.bin(name)


def main(argv):
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('tasks', metavar='<task>', nargs='*',
                      help="the name of tasks to run (default: all)")
  action_group = parser.add_mutually_exclusive_group()
  action_group.add_argument('-d', '--describe', action="store_true")
  action_group.add_argument('-u', '--upgrade', action='store_true')
  action_group.add_argument('-r', '--remove', action='store_true')
  action_group.add_argument('--versions', action="store_true")
  verbosity_group = parser.add_mutually_exclusive_group()
  verbosity_group.add_argument('-v', '--verbose', action='store_true')
  args = parser.parse_args(argv)

  # Configure logger
  if args.verbose:
    log_level = logging.DEBUG
  else:
    log_level = logging.INFO
  logging.basicConfig(level=log_level, format="%(message)s")

  with open(os.path.expanduser('~/phd/experimental/system/dotfiles/dotfiles.py')) as f:
    exec(f.read(), globals(), locals())

  # Get the list of tasks to run
  logging.debug("creating tasks list ...")
  queue = GetTasksToRun(args.tasks)
  done = set()
  ntasks = len(queue)

  fmt_bld, fmt_end, fmt_red = Colors.BOLD, Colors.END, Colors.RED

  # --describe flag prints a description of the work to be done:
  platform = GetPlatform()
  if args.describe:
    msg = ("There are {fmt_bld}{ntasks}{fmt_end} tasks to run on {platform}:"
           .format(**vars()))
    logging.info(msg)
    for i, task_name in enumerate(queue):
      task = TASKS[task_name]
      j = i + 1
      desc = type(task).__name__
      msg = ("[{j:2d}/{ntasks:2d}]  {fmt_bld}{task_name}{fmt_end} ({desc})"
             .format(**vars()))
      logging.info(msg)
      # build a list of generated files
      for file in task.genfiles:
        logging.debug("    " + os.path.abspath(os.path.expanduser(file)))

    return 0

  # --versions flag prints the specific task versions:
  if args.versions:
    for i, task_name in enumerate(queue):
      task = TASKS[task_name]
      for name in sorted(task.versions.keys()):
        version = task.versions[name]
        logging.info("{task_name}:{name}=={version}".format(**vars()))
    return 0

  if args.upgrade:
    task_type = "upgrade"
  elif args.remove:
    task_type = "uninstall"
  else:
    task_type = "install"

  msg = ("Running {fmt_bld}{ntasks} {task_type}{fmt_end} tasks on {platform}:"
         .format(**vars()))
  logging.info(msg)

  # Run the tasks
  ctx = CallContext()
  errored = False
  try:
    for i, task_name in enumerate(queue):
      task = TASKS[task_name]

      j = i + 1
      msg = "[{j:2d}/{ntasks:2d}] {fmt_bld}{task_name}{fmt_end} ...".format(**vars())
      logging.info(msg)

      start_time = time.time()

      # Resolve and run install() method:
      task(ctx)
      done.add(task)

      # Ensure that genfiles have been generated:
      if task_type == "install":
        for file in task.genfiles:
          file = os.path.abspath(os.path.expanduser(file))
          logging.debug("assert exists: '{file}'".format(**vars()))
          if not (os.path.exists(file) or
                  shell_ok("sudo test -f '{file}'".format(**vars())) or
                  shell_ok("sudo test -d '{file}'".format(**vars()))):
            raise InvalidTaskError('genfile "{file}" not created'.format(**vars()))
      runtime = time.time() - start_time

      logging.debug("{task_name} task completed in {runtime:.3f}s".format(**vars()))
      sys.stdout.flush()
  except KeyboardInterrupt:
    logging.info("\ninterrupt")
    errored = True
  except Exception as e:
    e_name = type(e).__name__
    logging.error("{fmt_bld}{fmt_red}fatal error: {e_name}".format(**vars()))
    logging.error(str(e) + Colors.END)
    errored = True
    if logging.getLogger().level <= logging.DEBUG:
      raise
  finally:
    # Task teardowm
    logging.debug(Colors.BOLD + "Running teardowns" + Colors.END)
    for task in done:
      task.TearDown(ctx)

      # build a list of temporary files
      tmpfiles = getattr(task, "__tmpfiles__", [])
      tmpfiles += getattr(task, "__" + GetPlatform() + "_tmpfiles__", [])

      # remove any temporary files
      for file in tmpfiles:
        file = os.path.abspath(os.path.expanduser(file))
        if os.path.exists(file):
          logging.debug("rm {file}".format(**vars()))
          os.remove(file)

  return 1 if errored else 0


if __name__ == '__main__':
  main(sys.argv[1:])
