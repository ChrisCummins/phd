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

import json

def pprint(data):
    print(json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")))

def info(*args, **kwargs):
    print("[INFO  ]", *args, **kwargs)

def debug(*args, **kwargs):
    print("[DEBUG ]", *args, **kwargs)

def warn(*args, **kwargs):
    print("[WARN  ]", *args, **kwargs)

def error(*args, **kwargs):
    print("[ERROR ]", *args, **kwargs)

def fatal(*args, status=1, **kwargs):
    error("fatal:", *args, **kwargs)
    lab.exit(status)

def colourise(colour, *args):
    return str(colour + str(args) + Colours.RESET)

#############################
# Shell escape colour codes #
#############################
class Colours:
    RESET   = '\033[0m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    RED     = '\033[91m'

    @staticmethod
    def print(colour, *args, **kwargs):
        print(colour, end="")
        print(*args, end="")
        print(Colours.RESET, **kwargs)
