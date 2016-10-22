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
Logging interface.
"""
from __future__ import print_function
import json
import sys

import labm8 as lab


def colourise(colour, *args):
    return "".join([colour] + list(args) + [Colours.RESET])


def printf(colour, *args, **kwargs):
    string = colourise(colour, *args)
    print(string, **kwargs)


def pprint(data, **kwargs):
    print(json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")),
          **kwargs)


def info(*args, **kwargs):
    print("[INFO  ]", *args, **kwargs)


def debug(*args, **kwargs):
    print("[DEBUG ]", *args, **kwargs)


def warn(*args, **kwargs):
    print("[WARN  ]", *args, **kwargs)


def error(*args, **kwargs):
    print("[ERROR ]", *args, **kwargs)


def fatal(*args, **kwargs):
    returncode = kwargs.pop("status", 1)
    error("fatal:", *args, **kwargs)
    lab.exit(returncode)


def prof(*args, **kwargs):
    """
    Print a profiling message.

    Profiling messages are intended for printing runtime performance
    data. They are prefixed by the "PROF" title.

    Arguments:

        *args, **kwargs: Message payload.
    """
    print("[PROF  ]", *args, **kwargs)


class Colours:
    """
    Shell escape colour codes.
    """
    RESET   = '\033[0m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    RED     = '\033[91m'
