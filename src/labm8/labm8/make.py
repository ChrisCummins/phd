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

import labm8
from labm8 import fs
from labm8 import system


def make(name="all", dir=".", stdout=system.STDOUT, stderr=system.STDERR):
    """
    Run make clean.
    """
    fs.cd(dir)
    ret, out, err = system.run(["make", str(name)], timeout=180,
                               stdout=stdout, stderr=stderr)
    fs.cdpop()
    return ret, out, err


def clean(**kwargs):
    """
    Run make clean.
    """
    make("clean", **kwargs)
