# Copyright (C) 2015, 2016 Chris Cummins.
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


_GLOBAL_TIMER = "__global__"
_timers = {}


class Error(Exception):
    """
    Module-level error class.
    """
    pass


class TimerNameError(Error):
    """
    Thrown in case of timer name conflicts or lookup misses.
    """
    pass


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


def _add_timer(name):
    _timers[name] = time()


def _get_elapsed_time(name):
    t = _timers[name]
    return (time() - t) * 1000


def _stop_timer(name):
    elapsed = _get_elapsed_time(name)
    del _timers[name]
    return elapsed


def _new_timer_name():
    name = "{:08x}".format(random.randrange(16 ** 8))
    return name if not isrunning(name) else _new_timer_name()


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

        TimerNameError: If `unique' is true and a timer with
          the same name already exists.
    """
    name = name or _GLOBAL_TIMER

    if unique and isrunning(name):
        raise TimerNameError("A timer named '{}' already exists".format(name))

    _add_timer(name)


def new():
    """
    Create and start a uniquely named timer.

    Generates a timer name which is guaranteed to be unique, then
    starts it.

    Returns:

        str: Name of new timer.
    """
    name = _new_timer_name()
    start(name=name)
    return name


def stop(name=None, **kwargs):
    """
    Stop a timer.

    Arguments:

       name (str, optional): The name of the timer to stop. If no name
         is given, stop the global anonymous timer.

    Raises:

        TimerNameError: If the named timer does not exist.
    """
    name = name or _GLOBAL_TIMER

    if name not in _timers:
        if name == _GLOBAL_TIMER:
            raise Error("Global timer has not been started")
        else:
            raise TimerNameError("No timer named '{}'".format(name))

    elapsed = int(round(_stop_timer(name)))
    if name == _GLOBAL_TIMER:
        name = "Timer"

    io.prof("{}: {} ms".format(name, elapsed), **kwargs)


def end(*args, **kwargs):
    """
    Stop a timer, see stop().
    """
    stop(*args, **kwars)


def reset(name=None):
    """
    Reset a timer.

    Arguments:

        name (str, optional): The name of the timer to reset. If no
          name is given, reset the global anonymous timer.

    Raises:

        TimerNameError: If the named timer does not exist.
    """
    if name not in _timers:
        raise TimerNameError("No timer named '{}'".format(name))

    _add_timer(name)


def elapsed(name=None):
    name = name or _GLOBAL_TIMER

    if name not in _timers:
        if name == _GLOBAL_TIMER:
            raise Error("Global timer has not been started")
        else:
            raise TimerNameError("No timer named '{}'".format(name))

    return _get_elapsed_time(name)
