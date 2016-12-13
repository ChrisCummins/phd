#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
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
Command line interface to clgen.
"""
from __future__ import print_function

import argparse
import os
import sys

from argparse import RawTextHelpFormatter
from sys import exit

import clgen
from clgen import log


def print_version_and_exit():
    """
    Print the clgen version. This function does not return.
    """
    print("clgen ", clgen.version())
    exit(0)


class ArgumentParser(argparse.ArgumentParser):
    """
    CLgen specialized argument parser.

    Differs from python argparse.ArgumentParser in the following areas:
      * Adds an optional `--verbose` flag and initializes the logging engine.
      * Adds an optional `--version` flag which prints version information and
        quits.
      * Defaults to using raw formatting for help strings, which disables line
        wrapping and whitespace squeezing.
      * Appends author information to description.
    """
    def __init__(self, *args, **kwargs):
        """
        See python argparse.ArgumentParser.__init__().
        """
        # append author information to description
        description = kwargs.get("description", "")

        if len(description) and description[-1] != "\n":
            description += "\n"
        description += """
Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
<http://chriscummins.cc/clgen>"""

        kwargs["description"] = description.lstrip()

        # unless specified otherwise, use raw text formatter. This means
        # that whitespace isn't squeed and lines aren't wrapped
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = RawTextHelpFormatter

        # call built in ArgumentParser constructor.
        super(ArgumentParser, self).__init__(*args, **kwargs)

        # Add defualt arguments
        self.add_argument("--version", action="store_true",
                          help="show version information and exit")
        self.add_argument("-v", "--verbose", action="store_true",
                          help="increase output verbosity")

    def parse_args(self, args=None, namespace=None):
        """
        See python argparse.ArgumentParser.parse_args().
        """
        if args is None:
            args = sys.argv[1:]

        # --version option overrides the normal argument parsing process.
        if len(args) == 1 and args[0] == "--version":
            print_version_and_exit()

        ret = super(ArgumentParser, self).parse_args(args, namespace)

        if ret.version:
            print_version_and_exit()

        # set log level
        log.init(ret.verbose)

        return ret


def main(method, *args, **kwargs):
    """
    Runs the given method as the main entrypoint to a program.

    If an exception is thrown, print error message and exit.

    If environmental variable DEBUG=1, then exception is not caught.

    Args:
        method (function): Function to execute.
        *args (str): Arguments for method.
        **kwargs (dict): Keyword arguments for method.

    Returns:
        method(*args, **kwargs)
    """
    # if DEBUG var set, don't catch exceptions
    if os.environ.get("DEBUG", None):
        return method(*args, **kwargs)

    try:
        return method(*args, **kwargs)
    except clgen.UserError as e:
        log.fatal(e, "(" + type(e).__name__  + ")")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\nkeyboard interrupt, terminating", file=sys.stderr)
        log.exit(1)
    except Exception as e:
        # TODO: Print limited stack trace
        log.fatal(e, "(" + type(e).__name__  + ")",
                  "\n\nPlease report bugs at <"
                  "https://github.com/ChrisCummins/clgen/issues>")
