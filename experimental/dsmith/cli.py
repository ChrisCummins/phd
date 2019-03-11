#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
import cProfile
import inspect
import logging
import os
import sys
import traceback
from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter
from typing import List

from experimental import dsmith
from experimental.dsmith import Colors
from experimental.dsmith.repl import repl, run_command
from labm8 import fs, prof


__help_epilog__ = """
Copyright (C) 2017 Chris Cummins <chrisc.101@gmail.com>.
<https://github.com/ChrisCummins/dsmith/>
"""


def getself(func):
  """ decorator to pass function as first argument to function """

  def wrapper(*args, **kwargs):
    return func(func, *args, **kwargs)

  return wrapper


def run(method, *args, **kwargs):
  """
  Runs the given method as the main entrypoint to a program.

  If an exception is thrown, print error message and exit.

  If environmental variable DEBUG=1, then exception is not caught.

  Arguments:
      method (function): Function to execute.
      *args: Arguments for method.
      **kwargs: Keyword arguments for method.

  Returns:
      method(*args, **kwargs): Method return value.
  """

  def _user_message(exception):
    logging.critical("""\
💩 Fatal error!
{err} ({type})

Please report bugs at <https://github.com/ChrisCummins/dsmith/issues>\
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

    trace = reversed(traceback.extract_tb(tb, limit=NUM_ROWS + 1)[1:])
    message = "\n".join(_msg(*r) for r in enumerate(trace))

    logging.critical("""\
💩 Fatal error!
{err} ({type})

  stacktrace:
{stack_trace}

Please report bugs at <https://github.com/ChrisCummins/dsmith/issues>\
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
      return method(*args, **kwargs)

    if prof.is_enabled() and logging.is_verbose():
      return cProfile.runctx('runctx()', None, locals(), sort='tottime')
    else:
      return runctx()
  except dsmith.UserError as err:
    logging.critical(err, "(" + type(err).__name__ + ")")
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
def main(self, args: List[str] = sys.argv[1:]):
  """
  Compiler fuzzing through deep learning.
  """
  parser = ArgumentParser(
      prog="dsmith",
      description=inspect.getdoc(self),
      epilog=__help_epilog__,
      formatter_class=RawDescriptionHelpFormatter)

  parser.add_argument(
      "--config",
      metavar="<path>",
      type=FileType("r"),
      dest="rc_path",
      help=f"path to configuration file (default: '{dsmith.RC_PATH}')")
  parser.add_argument(
      "-v", "--verbose", action="store_true", help="increase output verbosity")
  parser.add_argument(
      "--debug", action="store_true", help="debugging output verbosity")
  parser.add_argument(
      "--db-debug",
      action="store_true",
      help="additional database debugging output")
  parser.add_argument(
      "--version",
      action="store_true",
      help="show version information and exit")
  parser.add_argument(
      "--profile",
      action="store_true",
      help=("enable internal API profiling. When combined with --verbose, "
            "prints a complete profiling trace"))
  parser.add_argument(
      "command",
      metavar="<command>",
      nargs="*",
      help=("command to run. If not given, run an "
            "interactive prompt"))

  args = parser.parse_args(args)

  # set log level
  if args.debug:
    loglvl = logging.DEBUG
    os.environ["DEBUG"] = "1"

    # verbose stack traces. see: https://pymotw.com/2/cgitb/
    import cgitb
    cgitb.enable(format='text')
  elif args.verbose:
    loglvl = logging.INFO
  else:
    loglvl = logging.WARNING

  # set database log level
  if args.db_debug:
    os.environ["DB_DEBUG"] = "1"

  # configure logger
  logging.basicConfig(
      format='%(asctime)s [%(levelname)s] %(message)s', level=loglvl)

  # set profile option
  if args.profile:
    prof.enable()

  # load custom config:
  if args.rc_path:
    path = fs.abspath(args.rc_path.name)
    app.Log(2, f"loading configuration file '{Colors.BOLD}{path}{Colors.END}'")
    dsmith.init_globals(args.rc_path.name)

  # options whch override the normal argument parsing process.
  if args.version:
    print(dsmith.__version_str__)
  else:
    if len(args.command):
      # if a command was given, run it
      run_command(" ".join(args.command))
    else:
      # no command was given, fallback to interactive prompt
      repl()
