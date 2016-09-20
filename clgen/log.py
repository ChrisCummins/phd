"""
clgen logging interface.
"""
import logging

from sys import exit


def _fmt(msg, fmt_opts):
    """
    Format a message to a string.
    """
    assert(len(msg))
    sep = fmt_opts.get("sep", " ")
    return sep.join([str(x) for x in msg])


def init(verbose=False):
    """
    Initialiaze the logging engine.

    Arguments:
        verbose (bool, optional keyword): If True, print debug() messages.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def debug(*msg, **opts):
    """
    Debug message.

    If executing verbosely, prints the given message to stderr. To execute
    verbosely, intialize logging engine using log.init(verbose=True).

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
    """
    logging.debug(_fmt(msg, opts))


def info(*msg, **opts):
    """
    Info message.

    Prints the given message to stderr.

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
    """
    logging.info(_fmt(msg, opts))


def warning(*msg, **opts):
    """
    Warning message.

    Prints the given message to stderr prefixed with "warning: ".

    Arguments:
        *msg (sequence): Message to print.
        sep (str, optional keyword): Message component separator.
    """
    logging.warning("warning: " + _fmt(msg, opts))


def error(*msg, **opts):
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
