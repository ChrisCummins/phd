"""Internal logging library implementation."""
import fnmatch
import functools
import sys

from absl import flags as absl_flags
from absl import logging as absl_logging

FLAGS = absl_flags.FLAGS

absl_flags.DEFINE_list(
    'vmodule', [],
    "Per-module verbose level. The argument has to contain a comma-separated "
    "list of <module name>=<log level>. <module name> is a glob pattern (e.g., "
    "gfs* for all modules whose name starts with \"gfs\"), matched against the "
    "filename base (that is, name ignoring .py). <log level> overrides any "
    "value given by --v.")

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


def GetCallingModuleName():
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
def GetModuleVerbosity(module: str) -> int:
  """Return the verbosity level for the given module."""
  module_basename = module.split('.')[-1]
  for module_glob, level in ModuleGlob():
    if fnmatch.fnmatch(module_basename, module_glob):
      return level

  return absl_logging.get_verbosity() + 1


# Skip this function when determining the calling module and line number for
# logging.
@absl_logging.skip_log_prefix
def Log(calling_module_name: str, level: int, msg, *args, **kwargs):
  """Logs a message at the given level.

  Per-module verbose level. The argument has to contain a comma-separated
  list of <module name>=<log level>. <module name> is a glob pattern (e.g., "
    "gfs* for all modules whose name starts with \"gfs\"), matched against the "
    "filename base (that is, name ignoring .py). <log level> overrides any "
    "value given by --v."
  """
  module_level = GetModuleVerbosity(calling_module_name)
  if level <= module_level:
    absl_logging.info(msg, *args, **kwargs)


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
