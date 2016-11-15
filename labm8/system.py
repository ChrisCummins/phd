# Copyright (C) 2015, 2016 Chris Cummins.
#
# This file is part of labm8.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Utilies for grokking the underlying system.

Variables:
  * `HOSTNAME` (str) System hostname.
  * `USERNAME` (str) Username.
  * `UID` (int) User ID.
  * `PID` (int) Process ID.
"""

from __future__ import print_function

import getpass
import os
import socket
import sys
import subprocess
import threading

from sys import platform

import labm8 as lab
from labm8 import fs
from labm8 import io

HOSTNAME = socket.gethostname()
USERNAME = getpass.getuser()
UID = os.getuid()
PID = os.getpid()

argv = sys.argv
STDOUT = sys.stdout
STDERR = sys.stderr
PIPE = subprocess.PIPE


class Error(Exception):
    pass


class SubprocessError(Error):
    """
    Error thrown if a subprocess fails.
    """
    pass


class CommandNotFoundError(Exception):
    """
    Error thrown a system command is not found.
    """
    pass


class ScpError(Error):
    """
    Error thrown if scp file transfer fails.
    """
    def __init__(self, stdout, stderr):
        """
        Construct an ScpError.

        Arguments:

            stdout (str): Captured stdout of scp subprocess.
            stderr (str): Captured stderr of scp subprocess.
        """
        self.out = stdout
        self.err = stderr

    def __repr__(self):
        return self.out + "\n" + self.err

    def __str__(self):
        return self.__repr__()


class Subprocess(object):
    """
    Subprocess abstraction.

    Wrapper around subprocess.Popen() which provides the ability to
    force a timeout after a number of seconds have elapsed.
    """
    def __init__(self, cmd, shell=False,
                 stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE,
                 decode_out=True):
        """
        Create a new subprocess.
        """
        self.cmd = cmd
        self.process = None
        self.stdout = None
        self.stderr = None
        self.shell = shell
        self.decode_out = decode_out

        self.stdout_dest = stdout
        self.stderr_dest = stderr

    def run(self, timeout=-1):
        """
        Run the subprocess.

        Arguments:
            timeout (optional) If a positive real value, then timout after
                the given number of seconds.

        Raises:
            SubprocessError If subprocess has not completed after "timeout"
                seconds.
        """
        def target():
            self.process = subprocess.Popen(self.cmd,
                                            stdout=self.stdout_dest,
                                            stderr=self.stderr_dest,
                                            shell=self.shell)
            stdout, stderr = self.process.communicate()

            # Decode output if the user wants, and if there is any.
            if self.decode_out:
                if stdout:
                    self.stdout = stdout.decode("utf-8")
                if stderr:
                    self.stderr = stderr.decode("utf-8")

        thread = threading.Thread(target=target)
        thread.start()

        if timeout > 0:
            thread.join(timeout)
            if thread.is_alive():
                self.process.terminate()
                thread.join()
                raise SubprocessError(("Reached timeout after {t} seconds"
                                       .format(t=timeout)))
        else:
            thread.join()

        return self.process.returncode, self.stdout, self.stderr


def is_linux():
    return platform == "linux" or platform == "linux2"


def is_mac():
    return platform == "darwin"


def is_windows():
    return platform == "win32"


def run(command, num_retries=1, timeout=-1, **kwargs):
    """
    Run a command with optional timeout and retries.

    Provides a convenience method for executing a subprocess with
    additional error handling.

    Arguments:
        command (list of str): The command to execute.
        num_retries (int, optional): If the subprocess fails, the number of
          attempts to execute it before failing.
        timeout (float, optional): If positive, the number of seconds to wait
          for subprocess completion before failing.
        **kwargs: Additional args to pass to Subprocess.__init__()

    Returns:
        Tuple of (int, str, str): Where the variables represent
        (exit status, stdout, stderr).

    Raises:
        SubprocessError: If the command fails after the given number of
          retries.
    """
    last_error = None
    for _ in range(num_retries):
        try:
            process = Subprocess(command, **kwargs)
            return process.run(timeout)
        except Exception as err:
            last_error = err

    raise last_error


def sed(match, replacement, path, modifiers=""):
    """
    Perform sed text substitution.
    """
    cmd = "sed -r -i 's/%s/%s/%s' %s" % (match, replacement, modifiers, path)

    process = Subprocess(cmd, shell=True)
    ret, out, err = process.run(timeout=60)
    if ret:
        raise SubprocessError("Sed command failed!")


def echo(*args, **kwargs):
    """
    Write a message to a file.

    Arguments:
        args A list of arguments which make up the message. The last argument
            is the path to the file to write to.
    """
    msg = args[:-1]
    path = fs.path(args[-1])
    append = kwargs.pop("append", False)

    if append:
        with open(path, "a") as file:
            print(*msg, file=file, **kwargs)
    else:
        with open(fs.path(path), "w") as file:
            print(*msg, file=file, **kwargs)


def which(program, path=None):
    """
    Returns the full path of shell commands.

    Replicates the functionality of system which (1) command. Looks
    for the named program in the directories indicated in the $PATH
    environment variable, and returns the full path if found.

    Examples:

        >>> system.which("ls")
        "/bin/ls"

        >>> system.which("/bin/ls")
        "/bin/ls"

        >>> system.which("not-a-real-command")
        None

        >>> system.which("ls", path=("/usr/bin", "/bin"))
        "/bin/ls"

    Arguments:

        program (str): The name of the program to look for. Can
          be an absolute path.
        path (sequence of str, optional): A list of directories to
          look for the pgoram in. Default value is system $PATH.

    Returns:

       str: Full path to program if found, else None.
    """
    # If path is not given, read the $PATH environment variable.
    path = path or os.environ["PATH"].split(os.pathsep)
    abspath = True if os.path.split(program)[0] else False
    if abspath:
        if fs.isexe(program):
            return program
    else:
        for directory in path:
            # De-quote directories.
            directory = directory.strip('"')
            exe_file = os.path.join(directory, program)
            if fs.isexe(exe_file):
                return exe_file

    return None


def scp(host, src, dst, user=None, path=None):
    """
    Copy a file or directory from a remote location.

    A thin wrapper around the scp (1) system command.

    If the destination already exists, this will attempt to overwrite
    it.

    Arguments:

        host (str): name of the host
        src (str): path to the source file or directory.
        dst (str): path to the destination file or directory.
        user (str, optional): Alternative username for remote access.
          If not provided, the default scp behaviour is used.
        path (str, optional): Directory containing scp command. If not
          provided, attempt to locate scp using which().

    Raises:

        CommandNotFoundError: If scp binary not found.
        IOError: if transfer fails.
    """
    # Create the first argument.
    if user is None:
        arg = "{host}:{path}".format(host=host, path=src)
    else:
        arg = "{user}@{host}:{path}".format(user=user, host=host, path=src)

    # Get path to scp binary.
    scp_bin = which("scp", path=(path,))
    if scp_bin is None:
        raise CommandNotFoundError("Could not find scp in '{0}'".format(path))

    # Run system "scp" command.
    ret,out,err = run([scp_bin,
                       "-o", "StrictHostKeyChecking=no",
                       "-o", "UserKnownHostsFile=/dev/null",
                       arg, dst])

    # Check return code for error.
    if ret:
        raise ScpError(out, err)


def isprocess(pid, error=False):
    """
    Check that a process is running.

    Arguments:

        pid (int): Process ID to check.

    Returns:

        True if the process is running, else false.
    """
    try:
        # Don't worry folks, no processes are harmed in the making of
        # this system call:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
