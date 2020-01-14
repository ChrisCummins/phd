"""TODO."""
import os
import platform
import subprocess
import sys
from distutils import spawn

from experimental.util.dotfiles.implementation import io


def GetPlatform():
  distro = platform.linux_distribution()
  if not distro[0]:
    return {"darwin": "osx",}.get(sys.platform, sys.platform)
  else:
    return {"debian": "ubuntu",}.get(distro[0].lower(), distro[0].lower())


def Which(binary):
  """An UNIX-like implementation of the 'which' command."""
  return spawn.find_executable(binary)


def ShellCommand(*args):
  """ run a shell command and return its output. Raises CalledProcessError
      if fails """
  io.LogShellCommand(*args)
  p = subprocess.Popen(
    *args,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
  )
  stdout, _ = p.communicate()

  stdout = stdout.rstrip()
  io.LogShellOutput(stdout)

  if p.returncode:
    cmd = " ".join(args)
    msg = """\
Command '{cmd}' failed with returncode {p.returncode} and output:
{stdout}""".format(
      cmd=cmd, p=p, stdout=stdout
    )
    raise CalledProcessError(msg)
  else:
    return stdout


def CheckShellCommand(*args):
  """ run a shell command and return False if error """
  io.LogShellCommand(*args)
  try:
    subprocess.check_call(
      *args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    io.LogShellOutput("-> 0")
    return True
  except subprocess.CalledProcessError as e:
    io.LogShellOutput("-> " + str(e.returncode))
    return False


class CalledProcessError(EnvironmentError):
  pass


LINUX_DISTROS = ["debian", "ubuntu"]


def MakeSymlink(src, dst, sudo=False):
  src = os.path.expanduser(src)
  dst = os.path.expanduser(dst)

  if src.startswith("/"):
    src_abs = src
  else:
    src_abs = os.path.dirname(dst) + "/" + src

  # Symlink already exists
  use_sudo = "sudo -H " if sudo else ""
  if CheckShellCommand(
    "{use_sudo}test -f '{dst}'".format(**vars())
  ) or CheckShellCommand("{use_sudo}test -d '{dst}'".format(**vars())):
    linkdest = ShellCommand(
      "{use_sudo}readlink {dst}".format(**vars())
    ).rstrip()
    if linkdest.startswith("/"):
      linkdest_abs = linkdest
    else:
      linkdest_abs = os.path.dirname(dst) + "/" + linkdest
    if linkdest_abs == src_abs:
      return

  if not (
    CheckShellCommand("{use_sudo}test -f '{src_abs}'".format(**vars()))
    or CheckShellCommand("{use_sudo}test -d '{src_abs}'".format(**vars()))
  ):
    raise OSError("symlink source '{src}' does not exist".format(**vars()))
  # if CheckShellCommand("{use_sudo}test -d '{dst}'".format(**vars())):
  #     raise OSError("symlink destination '{dst}' is a directory".format(**vars()))

  # Make a backup of existing file:
  if CheckShellCommand(
    "{use_sudo}test -f '{dst}'".format(**vars())
  ) or CheckShellCommand("{use_sudo}test -d '{dst}'".format(**vars())):
    ShellCommand("{use_sudo}mv '{dst}' '{dst}'.backup".format(**vars()))

  # in case of broken symlink
  ShellCommand("{use_sudo}rm -f '{dst}'".format(**vars()))

  # Create the symlink:
  io.Print("Creating symlink {dst}".format(**vars()))
  ShellCommand("{use_sudo}ln -s '{src}' '{dst}'".format(**vars()))


def IsCompatible(a, b):
  """ return if platforms a and b are compatible """
  if b == "linux":
    return a in LINUX_DISTROS or a == "linux"
  else:
    return a == b
