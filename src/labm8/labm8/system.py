# Copyright (C) 2015 Chris Cummins.
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

"""
System module.

Variables:
  HOSTNAME System hostname.
  PID Process ID.
"""

from __future__ import print_function

import os
import socket
import sys
import subprocess
import threading

import labm8 as lab
from labm8 import fs
from labm8 import io

HOSTNAME = socket.gethostname()

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
            print(*msg, file=file)
    else:
        with open(fs.path(path), "w") as file:
            print(*msg, file=file)
