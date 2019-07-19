"""Unit tests for //labm8:dockerutil."""
import pathlib
import tempfile

from labm8 import dockerutil
from labm8 import test

FLAGS = test.FLAGS


def test_BazelPy3Image_CheckOutput():
  """Test output of image."""
  app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
  with app_image.RunContext() as ctx:
    output = ctx.CheckOutput([])
    assert output == 'Hello, world!\n'


def test_BazelPy3Image_CheckOutput_flags():
  """Test output of image with flags values."""
  app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
  with app_image.RunContext() as ctx:
    output = ctx.CheckOutput([], {'hello_to': 'Jason Isaacs'})
    assert output == 'Hello to Jason Isaacs!\n'


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
