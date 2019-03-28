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
import pathlib
import sys
from typing import Any, Callable, List, Optional, Union

from absl import app as absl_app
from absl import flags as absl_flags
from absl import logging as absl_logging

from config import build_info
from labm8.internal import flags_parsers
from labm8.internal import logging

FLAGS = absl_flags.FLAGS

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
      print("Copyright (C) 2014-2019 Chris Cummins <chrisc.101@gmail.com>")
      print(f"<{build_info.GetGithubCommitUrl()}>")
      sys.exit(0)
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


def GetVerbosity() -> int:
  """Get the verbosity level.

  This can be set per-module using --vmodule flag.
  """
  return logging.GetVerbosity(logging.GetCallingModuleName())


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
  calling_module = logging.GetCallingModuleName()
  logging.Log(calling_module, level, msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def LogIf(level: int, condition, msg, *args, **kwargs):
  if condition:
    calling_module = logging.GetCallingModuleName()
    logging.Log(calling_module, level, msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def Fatal(msg, *args, **kwargs):
  """Logs a fatal message."""
  logging.Fatal(msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def Error(msg, *args, **kwargs):
  """Logs an error message."""
  logging.Error(msg, *args, **kwargs)


@absl_logging.skip_log_prefix
def Warning(msg, *args, **kwargs):
  """Logs a warning message."""
  logging.Warning(msg, *args, **kwargs)


def FlushLogs():
  """Flushes all log files."""
  logging.FlushLogs()


# TODO(cec): Consider emoving DebugLogging() in favour of GetVerbosity().
def DebugLogging() -> bool:
  """Return whether debug logging is enabled."""
  return logging.DebugLogging()


def SetLogLevel(level: int) -> None:
  """Sets the logging verbosity.

  Causes all messages of level <= v to be logged, and all messages of level > v
  to be silently discarded.

  Args:
    level: the verbosity level as an integer.
  """
  logging.SetLogLevel(level)


# Flags functions.

# TODO(cec): Add flag_values argument to enable better testing.
# TODO(cec): Add validator callbacks.


def DEFINE_string(name: str,
                  default: Optional[str],
                  help: str,
                  required: bool = False,
                  validator: Callable[[str], bool] = None):
  """Registers a flag whose value can be any string."""
  absl_flags.DEFINE_string(
      name, default, help, module_name=logging.GetCallingModuleName())
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_integer(name: str,
                   default: Optional[int],
                   help: str,
                   required: bool = False,
                   lower_bound: Optional[int] = None,
                   upper_bound: Optional[int] = None,
                   validator: Callable[[int], bool] = None):
  """Registers a flag whose value must be an integer."""
  absl_flags.DEFINE_integer(
      name,
      default,
      help,
      module_name=logging.GetCallingModuleName(),
      lower_bound=lower_bound,
      upper_bound=upper_bound)
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_float(name: str,
                 default: Optional[float],
                 help: str,
                 required: bool = False,
                 lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None,
                 validator: Callable[[float], bool] = None):
  """Registers a flag whose value must be a float."""
  absl_flags.DEFINE_float(
      name,
      default,
      help,
      module_name=logging.GetCallingModuleName(),
      lower_bound=lower_bound,
      upper_bound=upper_bound)
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_boolean(name: str,
                   default: Optional[bool],
                   help: str,
                   required: bool = False,
                   validator: Callable[[bool], bool] = None):
  """Registers a flag whose value must be a boolean."""
  absl_flags.DEFINE_boolean(
      name, default, help, module_name=logging.GetCallingModuleName())
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_list(name: str,
                default: Optional[List[Any]],
                help: str,
                required: bool = False,
                validator: Callable[[List[Any]], bool] = None):
  """Registers a flag whose value must be a list."""
  absl_flags.DEFINE_list(
      name, default, help, module_name=logging.GetCallingModuleName())
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


# My custom flag types.


def DEFINE_input_path(name: str,
                      default: Union[None, str, pathlib.Path],
                      help: str,
                      is_dir: bool = False,
                      validator: Callable[[pathlib.Path], bool] = None):
  """Registers a flag whose value is an input path.

  An "input path" is a path to a file or directory that exists. The parsed value
  is a pathlib.Path instance. Flag parsing will fail if the value of this flag
  is not a path to an existing file or directory.

  Args:
    name: The name of the flag.
    default: The default value for the flag. While None is a legal value, it
      will fail during parsing - input paths are required flags.
    help: The help string.
    is_dir: If true, require the that the value be a directory. Else, require
      that the value be a file. Parsing will fail if this is not the case.
  """
  parser = flags_parsers.PathParser(must_exist=True, is_dir=is_dir)
  serializer = absl_flags.ArgumentSerializer()
  absl_flags.DEFINE(
      parser,
      name,
      default,
      help,
      absl_flags.FLAGS,
      serializer,
      module_name=logging.GetCallingModuleName())
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_output_path(name: str,
                       default: Union[None, str, pathlib.Path],
                       help: str,
                       is_dir: bool = False,
                       exist_ok: bool = True,
                       must_exist: bool = False,
                       validator: Callable[[pathlib.Path], bool] = None):
  """Registers a flag whose value is an output path.

  An "output path" is a path to a file or directory that may or may not already
  exist. The parsed value is a pathlib.Path instance. The idea is that this flag
  can be used to specify paths to files or directories that will be created
  during program execution. However, note that specifying an output path does
  not guarantee that the file will be produced.

  Args:
    name: The name of the flag.
    default: The default value for the flag. While None is a legal value, it
      will fail during parsing - output paths are required flags.
    help: The help string.
    is_dir: If true, require the that the value be a directory. Else, require
      that the value be a file. Parsing will fail if the path already exists and
      is of the incorrect type.
    exist_ok: If False, require that the path not exist, else parsing will fail.
    must_exist: If True, require that the path exists, else parsing will fail.
  """
  parser = flags_parsers.PathParser(
      must_exist=must_exist, exist_ok=exist_ok, is_dir=is_dir)
  serializer = absl_flags.ArgumentSerializer()
  absl_flags.DEFINE(
      parser,
      name,
      default,
      help,
      absl_flags.FLAGS,
      serializer,
      module_name=logging.GetCallingModuleName())
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_database(name: str,
                    database_class,
                    default: Optional[str],
                    help: str,
                    must_exist: bool = False,
                    validator: Callable[[Any], bool] = None):
  """Registers a flag whose value is a sqlutil.Database class.

  Unlike other DEFINE_* functions, the value produced by this flag is not an
  instance of the value, but a lambda that will instantiate a database of the
  requested type. This flag value must be called (with no arguments) in order to
  instantiate a database.

  Args:
    name: The name of the flag.
    database_class: The subclass of sqlutil.Database which is to be instantiated
      when this value is called, using the URL declared in 'default'.
    default: The default URL of the database. This is a required value.
    help: The help string.
    must_exist: If True, require that the database exists. Else, the database is
      created if it does not exist.
  """
  parser = flags_parsers.DatabaseParser(database_class, must_exist=must_exist)
  serializer = absl_flags.ArgumentSerializer()
  absl_flags.DEFINE(
      parser,
      name,
      default,
      help,
      absl_flags.FLAGS,
      serializer,
      module_name=logging.GetCallingModuleName())
  if validator:
    RegisterFlagValidator(name, validator)


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
