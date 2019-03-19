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
"""Python unit test main entry point.

This project uses pytest runner, with a handful of custom configuration options.
Use the Main() function as the entry point to your test files to run pytest
with the proper arguments.
"""
import contextlib
import inspect
import re
import sys
import pathlib
import tempfile
import typing

import pytest

from labm8 import app
from labm8 import fs

FLAGS = app.FLAGS

app.DEFINE_boolean('test_color', False, 'Colorize pytest output.')
app.DEFINE_boolean('test_skip_slow', True,
                   'Skip tests that have been marked slow.')
app.DEFINE_integer(
    'test_maxfail', 1,
    'The maximum number of tests that can fail before execution terminates. '
    'If --test_maxfail=0, all tests will execute.')
app.DEFINE_boolean('test_capture_output', True,
                   'Capture stdout and stderr during test execution.')
app.DEFINE_boolean(
    'test_print_durations', True,
    'Print the duration of the slowest tests at the end of execution. Use '
    '--test_durations to set the number of tests to print the durations of.')
app.DEFINE_integer(
    'test_durations', 3,
    'The number of slowest tests to print the durations of after execution. '
    'If --test_durations=0, the duration of all tests is printed.')
app.DEFINE_string(
    'test_coverage_data_dir', None,
    'Run tests with statement coverage and write coverage.py data files to '
    'this directory. The directory is created. Existing files are untouched.')


def GuessModuleUnderTest(file_path: str) -> typing.Optional[str]:
  # Strip everything up to the root of the project from the path.
  match = re.match(r'.+\.runfiles/phd/(.+)', file_path)
  if match:
    module = match.group(1)
    # Strip the _test.py suffix.
    module = module[:-len('_test.py')]
    # Convert path to fully qualified module.
    module = module.replace('/', '.')
    return module


@contextlib.contextmanager
def CoverageContext(file_path: str,
                    pytest_args: typing.List[str]) -> typing.List[str]:
  # Record coverage of module under test.
  if FLAGS.test_coverage_data_dir:
    module = GuessModuleUnderTest(file_path)
    if not module:
      app.Warning(
          'Not recording coverage - failed to determine module under test')
      yield pytest_args
      return

    with tempfile.TemporaryDirectory(prefix='phd_test_') as d:
      # Create a coverage.py config file.
      # See: https://coverage.readthedocs.io/en/coverage-4.3.4/config.html
      datadir = pathlib.Path(FLAGS.test_coverage_data_dir)
      datadir.mkdir(parents=True, exist_ok=True)
      config_path = f"{d}/converagerc"
      fs.Write(
          config_path, f"""
[run]
data_file = {datadir}/.coverage.{module}
parallel = true
""".encode('utf-8'))

      pytest_args += [
          f'--cov={module}',
          '--cov-config',
          config_path,
      ]
      yield pytest_args
  else:
    yield pytest_args


def RunPytestOnFileAndExit(file_path: str, argv: typing.List[str]):
  """Run pytest on a file and exit.

  This is invoked by absl.app.RunWithArgs(), and has access to absl flags.

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
    app.Fatal("File `%s` does not end in suffix _test.py", file_path)

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

  with CoverageContext(file_path, pytest_args) as pytest_args:
    app.Log(1, 'Running pytest with arguments: %s', pytest_args)
    ret = pytest.main(pytest_args)
  sys.exit(ret)


def Main():
  """Main entry point."""
  app.FLAGS(['argv[0]', '--vmodule=*=5'])

  # Get the file path of the calling function. This is used to identify the
  # script to run the tests of.
  frame = inspect.stack()[1]
  module = inspect.getmodule(frame[0])
  file_path = module.__file__

  app.RunWithArgs(lambda argv: RunPytestOnFileAndExit(file_path, argv))
