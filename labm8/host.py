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
import os
import socket
import subprocess

import labm8 as lab
from labm8 import io

class Error(Exception):
    pass

class SubprocessError(Error):
    """
    Error thrown if a subprocess fails.
    """
    pass

_HOSTNAME = None
def name():
    """
    Return the system hostname.
    """
    global _HOSTNAME

    if _HOSTNAME == None:
        _HOSTNAME = socket.gethostname()
        return name()
    else:
        return _HOSTNAME

_PID = None
def pid():
    """
    Return the current process ID.
    """
    global _PID
    if _PID == None:
        _PID = os.getpid()
        return pid()
    else:
        return _PID

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



def system(args, out=None, exit_on_error=True):
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
