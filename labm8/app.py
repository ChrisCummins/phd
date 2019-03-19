# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A wrapper around absl's app module.

See: <https://github.com/abseil/abseil-py>
"""
import fnmatch
import functools
import sys
from typing import Any, Callable, List, Optional

from absl import app as absl_app
from absl import flags as absl_flags
from absl import logging as absl_logging

from config import build_info

FLAGS = absl_flags.FLAGS

absl_flags.DEFINE_list(
    'vmodule', [],
    "Per-module verbose level. The argument has to contain a comma-separated "
    "list of <module name>=<log level>. <module name> is a glob pattern (e.g., "
    "gfs* for all modules whose name starts with \"gfs\"), matched against the "
    "filename base (that is, name ignoring .py). <log level> overrides any "
    "value given by --v.")
absl_flags.DEFINE_boolean('version', False,
                          'Print version information and exit.')


class UsageError(absl_app.UsageError):
  """Exception raised when the arguments supplied by the user are invalid.
  Raise this when the arguments supplied are invalid from the point of
  view of the application. For example when two mutually exclusive
  flags have been supplied or when there are not enough non-flag
  arguments.
  """

  def __init__(self, message, exitcode=1):
    super(UsageError, self).__init__(message)
    self.exitcode = exitcode


def AssertOrRaise(stmt: bool, exception: Exception, *exception_args,
                  **exception_kwargs) -> None:
  """If the statement is false, raise the given exception class."""
  if not stmt:
    raise exception(*exception_args, **exception_kwargs)


def RunWithArgs(main: Callable[[List[str]], None],
                argv: Optional[List[str]] = None):
  """Begin executing the program.

  Args:
    main: The main function to execute. It takes an single argument "argv",
      which is a list of command line arguments with parsed flags removed.
      If it returns an integer, it is used as the process's exit code.
    argv: A non-empty list of the command line arguments including program name,
      sys.argv is used if None.
  """

  def DoMain(argv):
    """Run the user-provided main method, with app-level arg handling."""
    if FLAGS.version:
      print(build_info.FormatShortBuildDescription())
      print(f"<{build_info.GetGithubCommitUrl()}>")
      sys.exit(1)
    main(argv)

  try:
    absl_app.run(DoMain, argv=argv)
  except KeyboardInterrupt:
    FlushLogs()
    sys.stdout.flush()
    sys.stderr.flush()
    print('keyboard interrupt')
    sys.exit(1)


def Run(main: Callable[[], None]):
  """Begin executing the program.

  Args:
    main: The main function to execute. It takes no arguments. If any command
    line arguments remain after flags parsing, an error is raised. If it
    returns an integer, it is used as the process's exit code.
  """

  def RunWithoutArgs(argv: List[str]):
    """Run the given function without arguments."""
    if len(argv) > 1:
      raise UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
    main()

  RunWithArgs(RunWithoutArgs)


# Logging functions.

# This is a set of module ids for the modules that disclaim key flags.
# This module is explicitly added to this set so that we never consider it to
# define key flag.
disclaim_module_ids = set([id(sys.modules[__name__])])


def get_module_object_and_name(globals_dict):
  """Returns the module that defines a global environment, and its name.
  Args:
    globals_dict: A dictionary that should correspond to an environment
      providing the values of the globals.
  Returns:
    _ModuleObjectAndName - pair of module object & module name.
    Returns (None, None) if the module could not be identified.
  """
  name = globals_dict.get('__name__', None)
  module = sys.modules.get(name, None)
  # Pick a more informative name for the main module.
  return module, (sys.argv[0] if name == '__main__' else name)


def _GetCallingModuleName():
  """Returns the module that's calling into this module.
  We generally use this function to get the name of the module calling a
  DEFINE_foo... function.
  Returns:
    The module name that called into this one.
  Raises:
    AssertionError: Raised when no calling module could be identified.
  """
  for depth in range(1, sys.getrecursionlimit()):
    # sys._getframe is the right thing to use here, as it's the best
    # way to walk up the call stack.
    globals_for_frame = sys._getframe(depth).f_globals  # pylint: disable=protected-access
    module, module_name = get_module_object_and_name(globals_for_frame)
    if id(module) not in disclaim_module_ids and module_name is not None:
      return module_name
  raise AssertionError('No module was found')


@functools.lru_cache(maxsize=1)
def ModuleGlob():
  return [(x.split('=')[0], int(x.split('=')[1])) for x in FLAGS.vmodule]


@functools.lru_cache(maxsize=128)
def _GetModuleVerbosity(module: str) -> int:
  """Return the verbosity level for the given module."""
  module_basename = module.split('.')[-1]
  for module_glob, level in ModuleGlob():
    if fnmatch.fnmatch(module_basename, module_glob):
      return level

  return absl_logging.get_verbosity() + 1


def GetVerbosity() -> int:
  """Get the verbosity level.

  This can be set per-module using --vmodule flag.
  """
  return _GetModuleVerbosity(_GetCallingModuleName())


# Skip this function when determining the calling module and line number for
# logging.
@absl_logging.skip_log_prefix
def Log(level: int, msg, *args, **kwargs):
  """Logs a message at the given level.

  Per-module verbose level. The argument has to contain a comma-separated
  list of <module name>=<log level>. <module name> is a glob pattern (e.g., "
    "gfs* for all modules whose name starts with \"gfs\"), matched against the "
    "filename base (that is, name ignoring .py). <log level> overrides any "
    "value given by --v."
  """
  calling_module = _GetCallingModuleName()
  module_level = _GetModuleVerbosity(calling_module)
  if level <= module_level:
    absl_logging.info(msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def LogIf(level: int, condition, msg, *args, **kwargs):
  if condition:
    Log(level, msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def Fatal(msg, *args, **kwargs):
  """Logs a fatal message."""
  absl_logging.fatal(msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def Error(msg, *args, **kwargs):
  """Logs an error message."""
  absl_logging.error(msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def Warning(msg, *args, **kwargs):
  """Logs a warning message."""
  absl_logging.warning(msg, *args, **kwargs)


def FlushLogs():
  """Flushes all log files."""
  absl_logging.flush()


def DebugLogging() -> bool:
  """Return whether debug logging is enabled."""
  return absl_logging.level_debug()


def SetLogLevel(level: int) -> None:
  """Sets the logging verbosity.

  Causes all messages of level <= v to be logged, and all messages of level > v
  to be silently discarded.

  Args:
    level: the verbosity level as an integer.
  """
  absl_logging.set_verbosity(level)


# Flags functions.

# TODO(cec): Implement DEFINE_path.
# TODO(cec): Add validator callbacks.
# TODO(cec): Add 'required' keyword to each flag.


def DEFINE_string(name, default, help):
  """Registers a flag whose value can be any string."""
  absl_flags.DEFINE_string(
      name, default, help, module_name=_GetCallingModuleName())


def DEFINE_integer(name, default, help, lower_bound=None, upper_bound=None):
  """Registers a flag whose value must be an integer."""
  absl_flags.DEFINE_integer(
      name,
      default,
      help,
      module_name=_GetCallingModuleName,
      lower_bound=lower_bound,
      upper_bound=upper_bound)


def DEFINE_float(name, default, help, lower_bound=None, upper_bound=None):
  """Registers a flag whose value must be a float."""
  absl_flags.DEFINE_float(
      name,
      default,
      help,
      module_name=_GetCallingModuleName,
      lower_bound=lower_bound,
      upper_bound=upper_bound)


def DEFINE_boolean(name, default, help):
  """Registers a flag whose value must be a boolean."""
  absl_flags.DEFINE_boolean(
      name, default, help, module_name=_GetCallingModuleName())


def DEFINE_list(name, default, help):
  """Registers a flag whose value must be a list."""
  absl_flags.DEFINE_list(
      name, default, help, module_name=_GetCallingModuleName())


def RegisterFlagValidator(flag_name: str,
                          checker: Callable[[Any], bool],
                          message: str = 'Flag validation failed'):
  """Adds a constraint, which will be enforced during program execution.

  The constraint is validated when flags are initially parsed, and after each
  change of the corresponding flag's value.

  Args:
    flag_name: str, name of the flag to be checked.
    checker: callable, a function to validate the flag.
        input - A single positional argument: The value of the corresponding
            flag (string, boolean, etc.  This value will be passed to checker
            by the library).
        output - bool, True if validator constraint is satisfied.
            If constraint is not satisfied, it should either return False or
            raise flags.ValidationError(desired_error_message).
    message: str, error text to be shown to the user if checker returns False.
        If checker raises flags.ValidationError, message from the raised
        error will be shown.

  Raises:
    AttributeError: Raised when flag_name is not registered as a valid flag
        name.
  """
  absl_flags.register_validator(flag_name, checker, message)
