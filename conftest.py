"""Repo-wide pytest configuration and test fixtures."""
import pathlib
import sys
import tempfile

import pytest
from absl import flags
from absl import logging


# *WARNING* Flags used in this file are not defined here! They are declared in
# //labm8:test.
FLAGS = flags.FLAGS


# Test fixtures.

@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  """A test fixture which yields a temporary directory."""
  with tempfile.TemporaryDirectory(prefix='phd_test_') as d:
    yield pathlib.Path(d)


# Pytest configuration.

# The names of platforms which can be used to mark tests.
PLATFORM_NAMES = set("darwin linux win32".split())


def pytest_collection_modifyitems(config, items):
  """A pytest hook to modify the configuration and items to run."""
  del config

  this_platform = sys.platform
  slow_skip_marker = pytest.mark.skip(reason='Use --notest_skip_slow to run')

  for item in items:
    # Skip tests if they been marked for an incompatible platform. To mark a
    # test for a platform, wrap the test function with a decorator. Example:
    #
    #   @pytest.mark.darwin
    #   def test_will_only_run_on_darwin():
    #     pass
    #
    # Platform decorators can be combined to support multiple platforms.
    supported_platforms = PLATFORM_NAMES.intersection(item.keywords)
    if supported_platforms and this_platform not in supported_platforms:
      skip_msg = f"Skipping `{item.name}` for platforms: {supported_platforms}"
      logging.info(skip_msg)
      item.add_marker(pytest.mark.skip(reason=skip_msg))
      continue

    # Skip tests that have been marked slow unless --notest_skip_slow. To mark
    # a test as slow, wrap the test function with a decorator. Example:
    #
    #   @pytest.mark.slow(reason='This takes a while')
    #   def test_long_running():
    #     ExpensiveTest()
    #
    # We could achieve the same effect by simple running with pytest with the
    # arguments `-m 'not slow'`, but skipping tests in this manner is silent.
    # Explicitly marking them as skipped, as done here, ensures that the test
    # name still appears in the test output, with a 'skipped' message. This is
    # useful for keeping track of how many tests in a file are *not* being run.
    if FLAGS.test_skip_slow and 'slow' in item.keywords:
      logging.info('Skipping `%s` because it is slow', item.name)
      item.add_marker(slow_skip_marker)
      continue
