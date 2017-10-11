#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Command line interface.
"""
import argparse
import cProfile
import inspect
import logging
import os
import sys
import traceback

from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter
from labm8 import jsonutil, fs, prof, types
from pathlib import Path
from sys import exit
from typing import BinaryIO, List, TextIO

import dsmith


# TODO: Append '<http://chriscummins.cc/dsmith>' to __help_epilog__
__help_epilog__ = """
Copyright (C) 2017 Chris Cummins <chrisc.101@gmail.com>.
"""

def getself(func):
    """ decorator to pass function as first argument to function """
    def wrapper(*args, **kwargs):
        return func(func, *args, **kwargs)
    return wrapper


class ReadableFilesOrDirectories(argparse.Action):
    """
    Adapted from @mgilson http://stackoverflow.com/a/11415816
    """

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        for path in values:
            if not os.path.isdir(path) and not os.path.isfile(path):
                raise argparse.ArgumentTypeError(
                    f"ReadableFilesOrDirectories:{path} not found")
            if not os.access(path, os.R_OK):
                raise argparse.ArgumentTypeError(
                    f"ReadableFilesOrDirectories:{path} is not readable")

        setattr(namespace, self.dest, [Path(path) for path in values])


def run(method, *args, **kwargs):
    """
    Runs the given method as the main entrypoint to a program.

    If an exception is thrown, print error message and exit.

    If environmental variable DEBUG=1, then exception is not caught.

    Parameters
    ----------
    method : function
        Function to execute.
    *args
        Arguments for method.
    **kwargs
        Keyword arguments for method.

    Returns
    -------
    method(*args, **kwargs)
    """
    def _user_message(exception):
        logging.critical("""\
{err} ({type})

Please report bugs to <chrisc.101@gmail.com>\
""".format(err=e, type=type(e).__name__))
        sys.exit(1)

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

        logging.critical("""\
{err} ({type})

  stacktrace:
{stack_trace}

Please report bugs to <chrisc.101@gmail.com>\
""".format(err=e, type=type(e).__name__, stack_trace=message))
        sys.exit(1)

    # if DEBUG var set, don't catch exceptions
    if os.environ.get("DEBUG", None):
        # verbose stack traces. see: https://pymotw.com/2/cgitb/
        import cgitb
        cgitb.enable(format='text')

        return method(*args, **kwargs)

    try:
        def runctx():
            method(*args, **kwargs)

        if prof.is_enabled() and logging.is_verbose():
            return cProfile.runctx('runctx()', None, locals(), sort='tottime')
        else:
            return runctx()
    except dsmith.UserError as err:
        logging.critical(err, "(" + type(err).__name__  + ")")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\nkeyboard interrupt, terminating", file=sys.stderr)
        sys.exit(1)
    except dsmith.UserError as e:
        _user_message(e)
    except dsmith.Filesystem404 as e:
        _user_message(e)
    except Exception as e:
        _user_message_with_stacktrace(e)


@getself
def _register_test_parser(self, parent: ArgumentParser) -> None:
    """
    Run the test suite.
    """

    def _main(cache_path: bool, coveragerc_path: bool,
              coverage_path: bool) -> None:
        import dsmith.test

        if cache_path:
            print(dsmith.test.test_cache_path())
            sys.exit(0)
        elif coveragerc_path:
            print(dsmith.test.coveragerc_path())
            sys.exit(0)
        elif coverage_path:
            print(dsmith.test.coverage_report_path())
            sys.exit(0)

        sys.exit(dsmith.test.testsuite())

    parser = parent.add_parser("test", help="run the testsuite",
                               description=inspect.getdoc(self),
                               epilog=__help_epilog__)
    parser.set_defaults(dispatch_func=_main)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cache-path", action="store_true",
                       help="print path to test cache")
    group.add_argument("--coveragerc-path", action="store_true",
                       help="print path to coveragerc file")
    group.add_argument("--coverage-path", action="store_true",
                       help="print path to coverage file")


@getself
def main(self, args: List[str]=sys.argv[1:]):
    """
    Compiler fuzzing through deep learning.
    """
    parser = ArgumentParser(
        prog="dsmith",
        description=inspect.getdoc(self),
        epilog="""
For information about a specific command, run `dsmith <command> --help`.

""" + __help_epilog__,
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="increase output verbosity")
    parser.add_argument(
        "--debug", action="store_true",
        help="maximum output verbosity")
    parser.add_argument(
        "--version", action="store_true",
        help="show version information and exit")
    parser.add_argument(
        "--profile", action="store_true",
        help=("enable internal API profiling. When combined with --verbose, "
              "prints a complete profiling trace"))

    subparser = parser.add_subparsers(title="available commands")

    subparsers = [
        _register_test_parser
    ]

    for register_fn in subparsers:
        register_fn(subparser)

    args = parser.parse_args(args)

    # set log level
    if args.debug:
        loglvl = logging.DEBUG
        os.environ["DEBUG"] = "1"
    elif args.verbose:
        loglvl = logging.INFO
    else:
        loglvl = logging.WARNING

    # configure logger
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=loglvl)

    # set profile option
    if args.profile:
        prof.enable()

    # options whch override the normal argument parsing process.
    if args.version:
        version = dsmith.version()
        print(f"dsmith {version} made with \033[1;31m♥\033[0;0m by "
              "Chris Cummins <chrisc.101@gmail.com>.")
    else:
        # strip the arguments from the top-level parser
        dispatch_func = args.dispatch_func
        opts = vars(args)
        del opts["version"]
        del opts["verbose"]
        del opts["debug"]
        del opts["profile"]
        del opts["dispatch_func"]

        # TODO: Handle case where no argument provided: if not len(opts):

        run(dispatch_func, **opts)
