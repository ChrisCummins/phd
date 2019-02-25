#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import errno
import hashlib
import inspect
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys

from distutils.spawn import find_executable


def get_platform():
  distro = platform.linux_distribution()
  if not distro[0]:
    return {
        "darwin": "osx",
    }.get(sys.platform, sys.platform)
  else:
    return {
        "debian": "ubuntu",
    }.get(distro[0].lower(), distro[0].lower())


class CalledProcessError(Exception):
  pass


def _log_shell(*args):
  if logging.getLogger().level <= logging.INFO:
    logging.debug(Colors.PURPLE + "    $ " + "".join(*args) + Colors.END)


def _log_shell_output(stdout):
  if logging.getLogger().level <= logging.INFO and len(stdout):
    indented = '    ' + '\n    '.join(stdout.rstrip().split('\n'))
    logging.debug(Colors.YELLOW + indented + Colors.END)


def shell(*args):
  """ run a shell command and return its output. Raises CalledProcessError
      if fails """
  _log_shell(*args)
  p = subprocess.Popen(
      *args,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      universal_newlines=True)
  stdout, _ = p.communicate()

  stdout = stdout.rstrip()
  _log_shell_output(stdout)

  if p.returncode:
    cmd = " ".join(args)
    msg = ("""\
Command '{cmd}' failed with returncode {p.returncode} and output:
{stdout}""".format(**vars()))
    raise CalledProcessError(msg)
  else:
    return stdout


def shell_ok(cmd):
  """ run a shell command and return False if error """
  _log_shell(cmd)
  try:
    subprocess.check_call(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _log_shell_output("-> 0")
    return True
  except subprocess.CalledProcessError as e:
    _log_shell_output("-> " + str(e.returncode))
    return False


# Run the configure script and read the resulting config.json file.
_c1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configure")
_c2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
shell(_c1)
with open(_c2) as infile:
  _CFG = json.loads(infile.read())

DOTFILES = _CFG["dotfiles"]
PRIVATE = _CFG["private"]
APPLE_ID = _CFG["apple_id"]
EXCLUDES = _CFG.get("excludes", [])
IS_TRAVIS_CI = os.environ.get("TRAVIS", False)

LINUX_DISTROS = ['debian', 'ubuntu']

PLATFORM = get_platform()
HOSTNAME = socket.gethostname()


def merge_dicts(a, b):
  """ returns a copy of 'a' with values updated by 'b' """
  dst = a.copy()
  dst.update(b)
  return dst


class Task(object):
  """
  A Task is a unit of work.

  Attributes:
    __platforms__ (List[str], optional): A list of platforms which the
        task may be run on. Any platform not in this list will not
        execute the task.
    __hosts__ (List[str], optional): A list of hostnames which the task
        may be run on. If this list is not present or is empty, all hosts
        are whitelisted.
    __deps__ (List[str], optional): A list of task classes which must be
        executed before the task may be run.
    __reqs__ (List[Callable], optional): A list of callable functions
        to determine if the task may be run.
    __<platform>_deps__ (List[Task], optional): A list of platform-specific
        tasks which must be executed before the task may be run.
    __genfiles__ (List[str], optional): A list of files generated during
        execution of this task.
    __<platform>_genfiles__ (List[str], optional): A platform-specific list
        of files generated during execution of this task.
    __tmpfiles__ (List[str], optional): A list of temporary files generated
        during execution of this task. These files are automatically
        removed after execution.
    __<platform>_tmpfiles__ (List[str], optional): A list of
        platform-specific temporary files generated during execution of
        this task. These files are automatically removed after execution.
    __<platform>_versions__ (Dict[str, str], optional): A dictionary mapping
        installed specific versions of packages.

  Methods:
    install():
    install_<platform>():
    uninstall():
    uninstall_<platform>():
    upgrade():
    upgrade_<platform>():
    teardown():
    teardown_<platform>():
  """
  __platforms__ = []
  __hosts__ = []
  __deps__ = []
  __reqs__ = []
  __genfiles__ = []
  __tmpfiles__ = []

  def setup(self):
    pass

  def teardown(self):
    pass

  def install(self):
    pass

  def upgrade(self):
    pass

  def uninstall(self):
    pass

  def __eq__(self, a):
    return type(self).__name__ == type(a).__name__

  def __repr__(self):
    return type(self).__name__

  @property
  def genfiles(self):
    """ return list of genfiles """
    ret = []
    if hasattr(self, "__genfiles__"):
      ret += self.__genfiles__

    if hasattr(self, "__" + get_platform() + "_genfiles__"):
      ret += getattr(self, "__" + get_platform() + "_genfiles__")

    return list(sorted(set(ret)))

  @property
  def deps(self):
    """ return list of dependencies """
    _deps = []
    if hasattr(self, "__deps__"):
      _deps += self.__deps__
    if hasattr(self, "__" + get_platform() + "_deps__"):
      _deps += getattr(self, "__" + get_platform() + "_deps__")

    return sorted(list(set(_deps)))

  @property
  def versions(task):
    """ return list of versions """
    _versions = dict()
    if hasattr(task, "__versions__"):
      _versions = merge_dicts(_versions, getattr(task, "__versions__"))
    if hasattr(task, "__" + get_platform() + "_version__"):
      _versions = merge_dicts(
          _versions, getattr(task, "__" + get_platform() + "_version__"))
    return versions


class InvalidTaskError(Exception):
  pass


class Colors:
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


def task_print(*msg, **kwargs):
  sep = kwargs.get("sep", " ")
  text = sep.join(msg)
  indented = "\n        > ".join(text.split("\n"))
  logging.info(Colors.GREEN + "        > " + indented + Colors.END)


def is_compatible(a, b):
  """ return if platforms a and b are compatible """
  if b == "linux":
    return a in LINUX_DISTROS or a == "linux"
  else:
    return a == b


def which(binary):
  return find_executable(binary)


def mkdir(path):
  path = os.path.abspath(os.path.expanduser(path))
  logging.debug("$ mkdir " + path)
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def symlink(src, dst, sudo=False):
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
    linkdest = shell("{use_sudo}readlink {dst}".format(**vars())).rstrip()
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
    shell("{use_sudo}mv '{dst}' '{dst}'.backup".format(**vars()))

  # in case of broken symlink
  shell("{use_sudo}rm -f '{dst}'".format(**vars()))

  # Create the symlink:
  task_print("Creating symlink {dst}".format(**vars()))
  shell("{use_sudo}ln -s '{src}' '{dst}'".format(**vars()))


def checksum_file(path):
  hash_fn = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_fn.update(chunk)
  return hash_fn.hexdigest()


def copy_file(src, dst):
  src = os.path.expanduser(src)
  dst = os.path.expanduser(dst)

  if not os.path.isfile(src):
    raise OSError("copy source '{src}' does not exist".format(**vars()))
  if os.path.isdir(dst):
    raise OSError("copy destination '{dst}' is a directory".format(**vars()))

  logging.debug("$ cp '{src}' '{dst}'".format(**vars()))

  src_checksum = checksum_file(src)
  dst_checksum = checksum_file(dst) if os.path.exists(dst) else None
  if src_checksum != dst_checksum:
    task_print("cp", src, dst)
    shutil.copyfile(src, dst)


def clone_git_repo(url,
                   destination,
                   version=None,
                   shallow=False,
                   recursive=True):
  """ clone a git repo, returns True if cloned """
  # Cannot set the version of a shallow clone.
  assert not (version and shallow)

  destination = os.path.abspath(os.path.expanduser(destination))
  cloned = False

  # clone repo if necessary
  if not os.path.isdir(destination):
    task_print("Cloning git repository to {destination}".format(**vars()))
    cmd = ['git clone "{url}" "{destination}"'.format(**vars())]
    if shallow:
      cmd.append('--depth 1')
    if recursive:
      cmd.append('--recursive')
    shell(' '.join(cmd))
    cloned = True

  if not os.path.isdir(os.path.join(destination, ".git")):
    raise OSError('directory "' + os.path.join(destination, ".git") +
                  '" does not exist')

  if version:
    # set revision
    pwd = os.getcwd()
    os.chdir(destination)
    target_hash = shell("git rev-parse {version} 2>/dev/null".format(**vars()))
    current_hash = shell("git rev-parse HEAD".format(**vars()))

    if current_hash != target_hash:
      shell("git fetch --all")
      shell("git reset --hard '{version}'".format(**vars()))

    os.chdir(pwd)

  return cloned


def github_repo(user, repo):
  """ get GitHub remote URL """
  if os.path.exists(os.path.expanduser("~/.ssh/id_rsa")):
    return "git@github.com:{user}/{repo}.git".format(**vars())
  else:
    return "https://github.com/{user}/{repo}.git".format(**vars())


def is_runnable_task(obj):
  """ returns true if object is a task for the current platform """
  # Check that object is a class and inherits from 'Task':
  if not (inspect.isclass(obj) and issubclass(obj, Task) and obj != Task):
    return False

  task = obj()
  # Check that task is compatible with platform:
  platforms = getattr(task, "__platforms__", [])
  if not any(is_compatible(PLATFORM, x) for x in platforms):
    msg = "skipping " + type(task).__name__ + " on platform " + PLATFORM
    logging.debug(msg)
    return False

  # Check that hostname is whitelisted (if whitelist is provided):
  hosts = getattr(task, "__hosts__", [])
  if hosts and HOSTNAME not in hosts:
    msg = "skipping " + type(task).__name__ + " on host " + HOSTNAME
    logging.debug(msg)
    return False

  # Check that task is not excluded:
  if type(task).__name__ in EXCLUDES:
    msg = "skipping " + type(task).__name__ + " as it is excluded"
    logging.debug(msg)
    return False

  # Check that task passes all req tests:
  reqs = getattr(task, "__reqs__", [])
  if reqs and not all(req() for req in reqs):
    msg = "skipping " + type(task).__name__ + ", failed req check"
    logging.debug(msg)
    return False

  return True


def get_task_method(task, method_name):
  """ resolve task method. First try and match <method>_<platform>(),
      then <method>() """
  fn = getattr(task, method_name + "_" + PLATFORM, None)
  if fn is None and PLATFORM in LINUX_DISTROS:
    fn = getattr(task, method_name + "_linux", None)
  if fn is None:
    fn = getattr(task, method_name, None)
  if fn is None:
    raise InvalidTaskError(
        "failed to resolve {method_name} method of Task {task}".format(
            **vars()))
  return fn


def usr_share(*components, **kwargs):
  """ fetch path to repo data """
  must_exist = kwargs.get("must_exist", True)
  path = os.path.join(DOTFILES, "usr", "share", *components)
  if must_exist and not os.path.exists(path):
    raise OSError(str(path) + " not found")
  return path


def get_version_name(version=None):
  """ return the name for a version """
  if version is None:
    version = shell('git rev-parse HEAD')
  with open('names.txt') as infile:
    names = infile.readlines()
  index = hash(version) % len(names)
  return names[index]
