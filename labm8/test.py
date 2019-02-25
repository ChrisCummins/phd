"""Python unit test main entry point.

This project uses pytest runner, with a handful of custom configuration options.
Use the Main() function as the entry point to your test files to run pytest
with the proper arguments.
"""
import inspect
import sys
import typing

import pytest
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_boolean('test_color', False, 'Colorize pytest output.')
flags.DEFINE_boolean('test_skip_slow', True,
                     'Skip tests that have been marked slow.')
flags.DEFINE_integer(
    'test_maxfail', 1,
    'The maximum number of tests that can fail before execution terminates. '
    'If --test_maxfail=0, all tests will execute.')
flags.DEFINE_boolean('test_capture_output', True,
                     'Capture stdout and stderr during test execution.')
flags.DEFINE_boolean(
    'test_print_durations', True,
    'Print the duration of the slowest tests at the end of execution. Use '
    '--test_durations to set the number of tests to print the durations of.')
flags.DEFINE_integer(
    'test_durations', 1,
    'The number of slowest tests to print the durations of after execution. '
    'If --test_durations=0, the duration of all tests is printed.')


def RunPytestOnFileAndExit(file_path: str, argv: typing.List[str]):
  """Run pytest on a file and exit.

  This is invoked by absl.app.run(), and has access to absl flags.

  This function does not return.

  Args:
    file_path: The path of the file to test.
    argv: Positional arguments not parsed by absl. No additional arguments are
      supported.
  """
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  # Test files must end with _test.py suffix. This is a code style choice, not
  # a hard requirement.
  if not file_path.endswith('_test.py'):
    logging.fatal("File `%s` does not end in suffix _test.py", file_path)

  # Assemble the arguments to run pytest with. Note that the //:conftest file
  # performs some additional configuration not captured here.
  pytest_args = [
      file_path,
      # Run pytest verbosely.
      '-vv',
  ]

  if FLAGS.test_color:
    pytest_args.append('--color=yes')

  if FLAGS.test_maxfail != 0:
    pytest_args.append(f'--maxfail={FLAGS.test_maxfail}')

  # Print the slowest test durations at the end of execution.
  if FLAGS.test_print_durations:
    pytest_args.append(f'--durations={FLAGS.test_durations}')

  # Capture stdout and stderr by default.
  if not FLAGS.test_capture_output:
    pytest_args.append('-s')

  logging.info('Running pytest with arguments: %s', pytest_args)
  sys.exit(pytest.main(pytest_args))


def Main():
  """Main entry point."""
  flags.FLAGS(['argv[0]', '-v=1'])

  # Get the file path of the calling function. This is used to identify the
  # script to run the tests of.
  frame = inspect.stack()[1]
  module = inspect.getmodule(frame[0])
  file_path = module.__file__

  app.run(lambda argv: RunPytestOnFileAndExit(file_path, argv))
