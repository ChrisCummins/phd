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
Command line interface to clgen.
"""
import cProfile
import argparse
import os
import sys
import traceback

from argparse import RawTextHelpFormatter
from labm8 import fs, prof
from sys import exit

import clgen
from clgen import log


def print_version_and_exit():
    """
    Print the clgen version. This function does not return.
    """
    version = clgen.version()
    print(f"clgen {version} made with \033[1;31m♥\033[0;0m by "
           "Chris Cummins <chrisc.101@gmail.com>.")
    exit(0)


class ArgumentParser(argparse.ArgumentParser):
    """
    CLgen specialized argument parser.

    Differs from python argparse.ArgumentParser in the following areas:
      * Adds an optional `--verbose` flag and initializes the logging engine.
      * Adds a `--debug` flag for more verbose crashes.
      * Adds a `--profile` flag for internal profiling.
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
Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
        self.add_argument("--debug", action="store_true",
                          help="in case of error, print debugging information")
        self.add_argument("--profile", action="store_true",
                          help="enable internal API profiling")

    def parse_args(self, args=sys.argv[1:], namespace=None):
        """
        See python argparse.ArgumentParser.parse_args().
        """
        # --version option overrides the normal argument parsing process.
        if "--version" in args:
            print_version_and_exit()

        # parse args normally
        args_ns = super(ArgumentParser, self).parse_args(args, namespace)

        # set log level
        log.init(args_ns.verbose)

        # set debug option
        if args_ns.debug:
            os.environ["DEBUG"] = "1"

        # set profile option
        if args_ns.profile:
            prof.enable()

        return args_ns


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
    def _user_message(exception):
        log.fatal("""\
{err} ({type})

Please report bugs at <https://github.com/ChrisCummins/clgen/issues>\
""".format(err=e, type=type(e).__name__))

    def _user_message_with_stacktrace(exception):
        # get limited stack trace
        def _msg(i, x):
            n = i + 1

            filename = fs.basename(x[0])
            lineno = x[1]
            fnname = x[2]

            loc = "{filename}:{lineno}".format(**vars())
            return "      #{n}  {loc: <18} {fnname}()".format(**vars())

        _, _, tb = sys.exc_info()
        NUM_ROWS = 5  # number of rows in traceback

        trace = reversed(traceback.extract_tb(tb, limit=NUM_ROWS+1)[1:])
        message = "\n".join(_msg(*r) for r in enumerate(trace))

        log.fatal("""\
{err} ({type})

  stacktrace:
{stack_trace}

Please report bugs at <https://github.com/ChrisCummins/clgen/issues>\
""".format(err=e, type=type(e).__name__, stack_trace=message))

    # if DEBUG var set, don't catch exceptions
    if os.environ.get("DEBUG", None):
        # verbose stack traces. see: https://pymotw.com/2/cgitb/
        import cgitb
        cgitb.enable(format='text')

        return method(*args, **kwargs)

    try:
        def run():
            method(*args, **kwargs)

        if prof.is_enabled():
            return cProfile.runctx('run()', None, locals(), sort='tottime')
        else:
            return run()
    except clgen.UserError as e:
        log.fatal(e, "(" + type(e).__name__  + ")")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\nkeyboard interrupt, terminating", file=sys.stderr)
        sys.exit(1)
    except clgen.UserError as e:
        _user_message(e)
    except clgen.File404 as e:
        _user_message(e)
    except Exception as e:
        _user_message_with_stacktrace(e)
