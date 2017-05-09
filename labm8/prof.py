# Copyright (C) 2015-2017 Chris Cummins.
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
Profiling API for timing critical paths in code.
"""
from __future__ import absolute_import

import random

from time import time

import labm8 as lab
from labm8 import fs
from labm8 import io


_GLOBAL_TIMER = "__default_timer__"
_timers = {}


def isrunning(name=None):
    """
    Check if a timer is running.

    Arguments:

        name (str, optional): The name of the timer to check.

    Returns:

        bool: True if timer is running, else False.
    """
    name = name or _GLOBAL_TIMER
    return name in _timers


def start(name=None, unique=False):
    """
    Start a new profiling timer.

    Arguments:

        name (str, optional): The name of the timer to create. If no
          name is given, the resulting timer is anonymous. There can
          only be one anonymous timer.
        unique (bool, optional): If true, then ensure that timer name
          is unique. This is to prevent accidentally resetting an
          existing timer.

    Raises:

        ValueError: If `unique' is true and a timer with
          the same name already exists.
    """
    name = name or _GLOBAL_TIMER

    if unique and name in _timers:
        raise ValueError("A timer named '{}' already exists".format(name))

    _timers[name] = time()


def stop(name=None, **kwargs):
    """
    Stop a timer.

    Arguments:

       name (str, optional): The name of the timer to stop. If no name
         is given, stop the global anonymous timer.

    Raises:

        LookupError: If the named timer does not exist.
    """
    quiet = kwargs.pop("quiet", False)
    name = name or _GLOBAL_TIMER

    if name not in _timers:
        if name == _GLOBAL_TIMER:
            raise Error("Global timer has not been started")
        else:
            raise LookupError("No timer named '{}'".format(name))

    elapsed = int(round((time() - _timers[name]) * 1000))
    del _timers[name]

    if name == _GLOBAL_TIMER:
        name = "Timer"

    if not quiet:
        io.prof("{}: {} ms".format(name, elapsed), **kwargs)


def reset(name=None):
    """
    Reset a timer.

    Arguments:

        name (str, optional): The name of the timer to reset. If no
          name is given, reset the global anonymous timer.

    Raises:

        LookupError: If the named timer does not exist.
    """
    name = name or _GLOBAL_TIMER

    if name not in _timers:
        raise LookupError("No timer named '{}'".format(name))

    _timers[name] = time()


def elapsed(name=None):
    """
    Reset a timer.

    Arguments:

        name (str, optional): The name of the timer to reset. If no
          name is given, reset the global anonymous timer.

    Returns:

        float: Elapsed time, in milliseconds.

    Raises:

        LookupError: If the named timer does not exist.
    """
    name = name or _GLOBAL_TIMER

    if name not in _timers:
        if name == _GLOBAL_TIMER:
            raise LookupError("Global timer has not been started")
        else:
            raise LookupError("No timer named '{}'".format(name))

    return (time() - _timers[name]) * 1000


def timers():
    for name in _timers:
        yield name, elapsed(name)
