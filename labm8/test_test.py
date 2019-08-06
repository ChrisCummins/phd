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
"""Unit tests for //labm8:test."""
import pathlib
import sys
import tempfile

import pytest

from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


# The //:conftest is included implicitly when you depend on //labm8:test.
def test_tempdir_fixture_directory_exists(tempdir: pathlib.Path):
  """Test that tempdir fixture returns a directory."""
  assert tempdir.is_dir()


def test_tempdir_fixture_directory_is_empty(tempdir: pathlib.Path):
  """Test that tempdir fixture returns an empty directory."""
  assert not list(tempdir.iterdir())


# Although the 'tempdir' fixture was defined in //:conftest, it can be
# overriden. This overiding fixture will be used for all of the tests in this
# file.
@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  """Override the tempdir fixture in //:conftest."""
  with tempfile.TemporaryDirectory(prefix='phd_fixture_override_') as d:
    yield pathlib.Path(d)


def test_tempdir_fixture_overriden(tempdir: pathlib.Path):
  """Test that the overriden test fixture is used, not the one in conftest."""
  assert tempdir.name.startswith('phd_fixture_override_')


@pytest.mark.slow(reason='This is an example')
def test_mark_slow():
  """A test that is skipped unless run with --notest_skip_slow."""
  pass


@pytest.mark.custom_marker
def test_custom_marker():
  """A test with a custom pytest marker that does nothing."""
  pass


@pytest.mark.win32
def test_that_only_runs_on_windows():
  pass


@pytest.mark.linux
def test_that_only_runs_on_linux():
  pass


@pytest.mark.darwin
def test_that_only_runs_on_darwin():
  pass


@pytest.mark.darwin
@pytest.mark.linux
def test_that_runs_on_linux_or_darwin():
  pass


def test_captured_stdout():
  """A test which prints to stdout."""
  print('This message is captured, unless run with --notest_capture_output')


def test_captured_stderr():
  """A test which prints to stderr."""
  print(
      'This message is captured, unless run with --notest_capture_output',
      file=sys.stderr,
  )


def test_captured_logging_info():
  """A test which prints to app.Log"""
  app.Log(1, 'This message is captured unless run with --notest_capture_output')


def test_captured_logging_debug():
  """A test which prints to app.Log"""
  app.Log(2, 'This message is captured unless run with --notest_capture_output')


def test_captured_logging_warning():
  """A test which prints to app.Warning"""
  app.Warning(
      'This message is captured unless run with --notest_capture_output',)


if __name__ == '__main__':
  test.Main()
