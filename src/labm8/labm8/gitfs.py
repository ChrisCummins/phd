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
import atexit
import os
import subprocess

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import system

# gitfs - File system abstraction.
#
# Abstracts file IO over a git file system.

DISK_WRITE_THRESHOLD = 50
REMOTES = {"origin": "master"}
MASTER_HOSTS = ["florence", "cec"]

_diskreads = 0
_diskread = set()

_diskwrites = 0
_diskwritten = set()

_cwd = fs.path(os.path.dirname(__file__))

def _commitandpush():
    global _diskwrites, _diskwritten

    # Don't commit from master hosts.
    if system.HOSTNAME in MASTER_HOSTS:
        return

    # Escape if we have nothing to do.
    if _diskwrites < 1:
        return

    io.printf(io.Colours.GREEN, "Commiting", len(_diskwritten), "files")

    for file in _diskwritten:
        fs.cd(os.path.dirname(file))
        subprocess.call(["git", "add", os.path.basename(file)])

    fs.cd(_cwd)
    subprocess.call(["git", "commit", "-m",
                     "{host}: Auto-bot commit".format(host=system.HOSTNAME)])
    subprocess.call(["git", "pull", "--rebase"])

    for remote in REMOTES:
        subprocess.call(["git", "push", remote, REMOTES[remote]])

    # Reset counters
    _diskwrites = 0
    _diskwritten = set()

# Register exit handler.
atexit.register(_commitandpush)

#
def markread(file):
    global _diskreads

    _diskreads += 1
    _diskread.add(file.name)
    return file

#
def markwrite(file):
    global _diskwrites

    _diskwrites += 1
    _diskwritten.add(file.name)

    if _diskwrites >= DISK_WRITE_THRESHOLD:
        _commitandpush()

    return file
