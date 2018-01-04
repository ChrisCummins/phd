from __future__ import print_function

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
        return distro[0].lower()


def _shell(*args, **kwargs):
    error = kwargs.get("error", True)
    stdout = kwargs.get("stdout", False)

    logging.debug("$ " + "".join(*args))
    if stdout:
        return subprocess.check_output(*args, shell=shell, universal_newlines=True,
                                       stderr=subprocess.PIPE)
    elif error:
        return subprocess.check_call(*args, shell=shell, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
    else:
        try:
            subprocess.check_call(*args, shell=shell, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False

def shell(*args):
    return _shell(*args)


def shell_ok(*args):
    return _shell(*args, error=False)


def shell_output(*args):
    return _shell(*args, stdout=True)


shell("./configure")
with open("config.json") as infile:
    _CFG = json.loads(infile.read())

DOTFILES = _CFG["dotfiles"]
PRIVATE = _CFG["private"]
APPLE_ID = _CFG["apple_id"]
IS_TRAVIS_CI = os.environ.get("TRAVIS", False)

LINUX_DISTROS = ['ubuntu']

PLATFORM = get_platform()
HOSTNAME = socket.gethostname()


class Task(object):
    """
    A Task is a unit of work.

    Attributes:
        __platforms__ (List[str], optional): A list of platforms which the
            task may be run on. Any platform not in this list will not
            execute the task.
        __deps__ (List[str], optional): A list of task classes which must be
            executed before the task may be run.
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
    __deps__ = []
    __genfiles__ = []
    __tmpfiles__ = []

    def setup(self):
        pass

    def run(self):
        """ """
        raise NotImplementedError

    def teardown(self):
        pass

    def upgrade(self):
        pass

    def uninstall(self):
        pass

    def __eq__(self, a):
        return type(self).__name__ == type(a).__name__

    def __repr__(self):
        return type(self).__name__


class InvalidTaskError(Exception): pass


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
    print(Colors.GREEN, "        > ", indented, Colors.END, sep="")


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
        linkdest = shell_output("{use_sudo}readlink {dst}".format(**vars())).rstrip()
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


def clone_git_repo(url, destination, version=None):
    """ clone a git repo, returns True if cloned """
    destination = os.path.abspath(os.path.expanduser(destination))
    cloned = False

    # clone repo if necessary
    if not os.path.isdir(destination):
        task_print("Cloning git repository to {destination}".format(**vars()))
        shell('git clone --recursive "{url}" "{destination}"'.format(**vars()))
        cloned = True

    if not os.path.isdir(os.path.join(destination, ".git")):
        raise OSError('directory "' + os.path.join(destination, ".git") +
                      '" does not exist')

    if version:
        # set revision
        pwd = os.getcwd()
        os.chdir(destination)
        target_hash = shell_output("git rev-parse {version} 2>/dev/null".format(**vars()))
        current_hash = shell_output("git rev-parse HEAD".format(**vars()))

        if current_hash != target_hash:
            shell("git fetch --all")
            shell("git reset --hard '{version}'".format(**vars()))

        os.chdir(pwd)

    return cloned


def is_runnable_task(obj):
    """ returns true if object is a task for the current platform """
    if not (inspect.isclass(obj) and  issubclass(obj, Task) and obj != Task):
        return False

    task = obj()
    platforms = getattr(task, "__platforms__", [])
    if not any(is_compatible(PLATFORM, x) for x in platforms):
        logging.debug("skipping " + type(task).__name__ + " on platform " + PLATFORM)
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
        raise InvalidTaskError("failed to resolve {method_name} method of Task {task}".format(**vars()))
    return fn


def get_task_deps(task):
    """ resolve list of dependencies for task """
    deps = []
    if hasattr(task, "__deps__"):
        deps += getattr(task, "__deps__")
    if hasattr(task, "__" + get_platform() + "_deps__"):
        deps += getattr(task, "__" + get_platform() + "_deps__")

    return sorted(list(set(deps)))


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
        version = shell_output('git rev-parse HEAD')
    with open('names.txt') as infile:
        names = infile.readlines()
    index = hash(version) % len(names)
    return names[index]
