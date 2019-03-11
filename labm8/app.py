"""A wrapper around absl's app module.

See: <https://github.com/abseil/abseil-py>
"""
import sys
from typing import Callable, List, Optional

from absl import app as absl_app
from absl import flags as absl_flags
from absl import logging as absl_logging

FLAGS = absl_flags.FLAGS


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
  try:
    absl_app.run(main, argv=argv)
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

  try:
    absl_app.run(main)
  except KeyboardInterrupt:
    FlushLogs()
    sys.stdout.flush()
    sys.stderr.flush()
    print('keyboard interrupt')
    sys.exit(1)


# Logging functions.

# TODO(cec): Emulate vmodule behavior.


def Fatal(msg, *args, **kwargs):
  """Logs a fatal message."""
  absl_logging.fatal(msg, *args, **kwargs)


def Error(msg, *args, **kwargs):
  """Logs an error message."""
  absl_logging.error(msg, *args, **kwargs)


def Warning(msg, *args, **kwargs):
  """Logs a warning message."""
  absl_logging.warning(msg, *args, **kwargs)


def Info(msg, *args, **kwargs):
  """Logs an info message."""
  absl_logging.info(msg, *args, **kwargs)


def Debug(msg, *args, **kwargs):
  """Logs a debug message."""
  absl_logging.debug(msg, *args, **kwargs)


def FlushLogs():
  """Flushes all log files."""
  absl_logging.flush()


def DebugLogging() -> bool:
  """Return whether debug logging is enabled."""
  return absl_logging.level_debug()


# Flags functions.

# TODO(cec): Implement DEFINE_path.
# TODO(cec): Add validator callbacks.
# TODO(cec): Add 'required' keyword to each flag.

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
    The module object that called into this one.
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
