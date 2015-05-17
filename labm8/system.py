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

import os
import socket
import subprocess

import labm8 as lab
from labm8 import fs
from labm8 import io

HOSTNAME = socket.gethostname()

PID = os.getpid()


class Error(Exception):
    pass


class SubprocessError(Error):
    """
    Error thrown if a subprocess fails.
    """
    pass


def check_output(args, shell=False, exit_on_error=True):
    """Run "args", returning stdout and stderr.

    If the process fails, raises a SubprocessError. If "exit_on_error"
    is True, execution halts.
    """
    try:
        output = subprocess.check_output(args, shell=shell,
                                         stderr=subprocess.STDOUT)
        return output.decode()
    except subprocess.CalledProcessError as err:
        msg = ("Subprocess '%s' failed with exit code '{}'"
               % err.cmd, err.returncode)

        if exit_on_error:
            io.error(err.output)
            io.fatal(msg, err.returncode)
        raise SubprocessError(msg)


def run(args, out=None, exit_on_error=True):
    """
    Run "args", redirecting stdout and stderr to "out". Returns exit
    status.
    """
    try:
        returncode = subprocess.call(args, stdout=out, stderr=out)
    except KeyboardInterrupt:
        print()
        io.fatal("Keyboard interrupt", status=0)
    if returncode and exit_on_error:
        io.fatal(args, status=returncode)
    return returncode


def sed(match, replacement, path, modifiers=""):
    """
    Perform sed text substitution.
    """
    cmd = "sed -r -i 's/%s/%s/%s' %s" % (match, replacement, modifiers, path)

    os.system(cmd)
