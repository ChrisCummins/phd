#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
clgen logging interface.
"""
import logging

from sys import exit


def _fmt(msg: str, fmt_opts: dict) -> str:
    """
    Format a message to a string.
    """
    assert(len(msg))
    sep = fmt_opts.get("sep", " ")
    return sep.join([str(x) for x in msg])


def init(verbose: bool=False) -> None:
    """
    Initialiaze the logging engine.

    Arguments:
        verbose (bool, optional keyword): If True, print debug() messages.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def is_verbose() -> bool:
    return logging.getLogger().getEffectiveLevel() == logging.DEBUG


def debug(*msg, **opts) -> None:
    """
    Debug message.

    If executing verbosely, prints the given message to stderr. To execute
    verbosely, intialize logging engine using log.init(verbose=True).

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
    """
    logging.debug(_fmt(msg, opts))


def verbose(*msg, **opts) -> None:
    """
    Calls debug().
    """
    debug(*msg, **opts)


def info(*msg, **opts) -> None:
    """
    Info message.

    Prints the given message to stderr.

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
    """
    logging.info(_fmt(msg, opts))


def warning(*msg, **opts) -> None:
    """
    Warning message.

    Prints the given message to stderr prefixed with "warning: ".

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
    """
    logging.warning("warning: " + _fmt(msg, opts))


def error(*msg, **opts) -> None:
    """
    Error message.

    Prints the given message to stderr prefixed with "error: ".

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
    """
    logging.error("error: " + _fmt(msg, opts))


def fatal(*msg, **opts):
    """
    Fatal error.

    Prints the given message to stderr prefixed with "fatal: ", then exists.
    This function does not return.

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
        ret (int, optional keyword): Value to exit with.
    """
    logging.error("fatal: " + _fmt(msg, opts))
    ret = opts.get("ret", 1)
    exit(ret)
