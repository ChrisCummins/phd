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
import labm8 as lab
import labm8.io

import os
import socket
import subprocess

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

def system(args, out=None, exit_on_error=False):
    """
    Run "args", redirecting stdout and stderr to "out". Returns exit
    status.
    """
    try:
        exitstatus = subprocess.call(args, stdout=out, stderr=out)
    except KeyboardInterrupt:
        print()
        lab.io.fatal("Keyboard interrupt", status=0)
    if exitstatus != 0 and exit_on_error:
        lab.io.fatal(*args, status=exitstatus)
    return exitstatus
