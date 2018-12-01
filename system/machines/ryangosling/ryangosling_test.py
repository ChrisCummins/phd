"""Black box tests for //system/machines/ryangosling:ryangosling.pbtxt.

These tests will NOT pass if run on one of my personal machines.
"""
import os

import pytest
import sys
import typing
from absl import app
from absl import flags

from lib.labm8 import bazelutil
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


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
