"""Unit tests for //labm8:dockerutil."""
import pathlib

from labm8 import dockerutil
from labm8 import test

FLAGS = test.FLAGS


def test_BazelPy3Image_CheckOutput():
  """Test output of image."""
  app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
  with app_image.RunConext() as ctx:
    output = ctx.CheckOutput([])
    assert output == 'Hello, world!\n'


def test_BazelPy3Image_CheckOutput_flags():
  """Test output of image with flags values."""
  app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
  with app_image.RunConext() as ctx:
    output = ctx.CheckOutput([], {'hello_to': 'Jason Isaacs'})
    assert output == 'Hello to Jason Isaacs!\n'


def test_BazelPy3Image_CheckCall_shared_volume(tempdir: pathlib.Path):
  """Test shared volume."""
  app_image = dockerutil.BazelPy3Image('labm8/test_data/basic_app')
  with app_image.RunConext() as ctx:
    ctx.CheckCall(['--create_file'], volumes={tempdir: '/tmp'})
  assert (tempdir / 'hello.txt').is_file()


if __name__ == '__main__':
  test.Main()
