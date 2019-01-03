"""Black box tests for //system/machines/ryangosling:ryangosling.pbtxt.

These tests will NOT pass if run on one of my personal machines.
"""
import os

import pytest
from absl import flags

from labm8 import bazelutil
from labm8 import test
from system.machines import machine
from system.machines import mirrored_directory


FLAGS = flags.FLAGS

_MACHINE_SPEC_PATH = bazelutil.DataPath(
    'phd/system/machines/ryangosling/ryangosling.pbtxt')


@pytest.fixture(scope='function')
def ryangosling() -> machine.Machine:
  return machine.Machine.FromFile(_MACHINE_SPEC_PATH)


def test_Ryangosling_mirrored_directories(ryangosling: machine.Machine):
  """Test that mirrored directories exists."""
  assert len(ryangosling.mirrored_directories) == 4


@pytest.mark.skipif(not os.path.isdir('/Volumes/Orange'),
                    reason='Orange drive not mounted')
def test_Ryangosling_photos(ryangosling: machine.Machine):
  """Test that mirrored directory exists."""
  d = ryangosling.MirroredDirectory('photos')
  assert isinstance(d, mirrored_directory.MirroredDirectory)
  assert d.RemoteExists()
  assert d.LocalExists()


def test_Ryangosling_music(ryangosling: machine.Machine):
  """Test that mirrored directory exists."""
  d = ryangosling.MirroredDirectory('music')
  assert isinstance(d, mirrored_directory.MirroredDirectory)
  assert d.RemoteExists()
  assert d.LocalExists()


def test_Ryangosling_movies(ryangosling: machine.Machine):
  """Test that mirrored directory exists."""
  d = ryangosling.MirroredDirectory('movies')
  assert isinstance(d, mirrored_directory.MirroredDirectory)
  assert d.RemoteExists()
  assert d.LocalExists()


def test_Ryangosling_tv(ryangosling: machine.Machine):
  """Test that mirrored directory exists."""
  d = ryangosling.MirroredDirectory('tv')
  assert isinstance(d, mirrored_directory.MirroredDirectory)
  assert d.RemoteExists()
  assert d.LocalExists()


if __name__ == '__main__':
  test.Main()
