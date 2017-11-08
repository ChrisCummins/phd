import errno
import inspect
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


DOTFILES = os.path.expanduser("~/.dotfiles")
PRIVATE = os.path.expanduser("~/Dropbox/Shared")

LINUX_DISTROS = ['ubuntu']

PLATFORM = get_platform()
HOSTNAME = socket.gethostname()


class Task(object):
    """
    A Task is a unit of work.

    Each task consists of a run() method

    {setup,run,teardown}_<platform>()

    Attributes:
        __platforms__ (List[str], optional): A list of platforms which the
            task may be run on. Any platform not in this list will not
            execute the task.
        __deps__ (List[Task], optional): A list of tasks which must be executed
            before this task may be run.
        __genfiles__ (List[str], optional): A list of permanent files generated
            during execution of this task.
        __tmpfiles__ (List[str], optional): A list of files generated during
            execution of this task.

    Methods:
        setup():
        run():
        teardown():
        run_osx()
    run()
    """
    def setup(self):
        pass

    def run(self):
        """ """
        raise NotImplementedError

    def teardown(self):
        pass

    def __repr__(self):
        return type(self).__name__


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

    # Create the symlink:
    shell("{use_sudo}ln -s '{src}' '{dst}'".format(**vars()))


def copy(src, dst):
    src = os.path.expanduser(src)
    dst = os.path.expanduser(dst)

    if not os.path.isfile(src):
        raise OSError("copy source '{src}' does not exist".format(**vars()))
    if os.path.isdir(dst):
        raise OSError("copy destination '{dst}' is a directory".format(**vars()))

    logging.debug("$ cp '{src}' '{dst}'".format(**vars()))
    shutil.copyfile(src, dst)


def clone_git_repo(url, destination, version):
    destination = os.path.abspath(os.path.expanduser(destination))

    # clone repo if necessary
    if not os.path.isdir(destination):
        shell('git clone --recursive "{url}" "{destination}"'.format(**vars()))

    if not os.path.isdir(os.path.join(destination, ".git")):
        raise OSError('directory "' + os.path.join(destination, ".git") +
                      '" does not exist')

    # set revision
    pwd = os.getcwd()
    os.chdir(destination)
    target_hash = shell_output("git rev-parse {version} 2>/dev/null".format(**vars()))
    current_hash = shell_output("git rev-parse HEAD".format(**vars()))

    if current_hash != target_hash:
        logging.info("setting repo version {destination} to {version}".format(**vars()))
        shell("git fetch --all")
        shell("git reset --hard '{version}'".format(**vars()))

    os.chdir(pwd)


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
        raise InvalidTaskError
    return fn
