"""Unit tests for //labm8:dockerutil."""
import pathlib
import tempfile

import pytest

from labm8 import dockerutil
from labm8 import system
from labm8 import test

FLAGS = test.FLAGS

if not system.which('docker'):
  # No tests will run if we don't have docker.
  MODULE_UNDER_TEST = None

# Annotation for tests that require 'docker' in the system PATH.
requires_docker = pytest.mark.skipif(
    not system.which('docker'),
    reason='docker binary not found in $PATH',
)


@requires_docker
def test_BazelPy3Image_CheckOutput():
  """Test output of image."""
  app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
  with app_image.RunContext() as ctx:
    output = ctx.CheckOutput([])
    assert output == 'Hello, world!\n'


@requires_docker
def test_BazelPy3Image_CheckOutput_flags():
  """Test output of image with flags values."""
  app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
  with app_image.RunContext() as ctx:
    output = ctx.CheckOutput([], {'hello_to': 'Jason Isaacs'})
    assert output == 'Hello to Jason Isaacs!\n'


@requires_docker
def test_BazelPy3Image_CheckCall_shared_volume():
  """Test shared volume."""
  # Force a temporary directory inside /tmp, since on macOS,
  # tempfile.TemporaryDirectory() can generate a directory outside of those
  # available to docker. See:
  # https://docs.docker.com/docker-for-mac/osxfs/#namespaces
  with tempfile.TemporaryDirectory(prefix='phd_dockerutil_', dir='/tmp') as d:
    tmpdir = pathlib.Path(d)
    app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
    with app_image.RunContext() as ctx:
      ctx.CheckCall(['--create_file'], volumes={tmpdir: '/tmp'})
    assert (tmpdir / 'hello.txt').is_file()


if __name__ == '__main__':
  test.Main()
