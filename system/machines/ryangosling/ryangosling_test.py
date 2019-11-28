"""Black box tests for //system/machines/ryangosling:ryangosling.pbtxt.

These tests will NOT pass if run on one of my personal machines.
"""
import os

import pytest

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import test
from system.machines import machine
from system.machines import mirrored_directory

FLAGS = app.FLAGS

MODULE_UNDER_TEST = "system.machines"

_MACHINE_SPEC_PATH = bazelutil.DataPath(
  "phd/system/machines/ryangosling/ryangosling.pbtxt"
)


@pytest.fixture(scope="function")
def ryangosling() -> machine.Machine:
  return machine.Machine.FromFile(_MACHINE_SPEC_PATH)


def test_Ryangosling_mirrored_directories(ryangosling: machine.Machine):
  """Test that mirrored directories exists."""
  # FRAGILE TEST: This may need to be updated whenever
  # //system/machines/ryangosling/ryangosling.pbtxt is modified!
  assert len(ryangosling.mirrored_directories) == 6


@pytest.mark.diana
@pytest.mark.florence
@pytest.mark.skipif(
  not os.path.isdir("/Volumes/Orange"), reason="Orange drive not found"
)
def test_Ryangosling_photos(ryangosling: machine.Machine):
  """Test that mirrored directory exists."""
  d = ryangosling.MirroredDirectory("photos")
  assert isinstance(d, mirrored_directory.MirroredDirectory)
  assert d.RemoteExists()
  assert d.LocalExists()


@pytest.mark.diana
def test_Ryangosling_photos(ryangosling: machine.Machine):
  """Test that mirrored directory exists."""
  d = ryangosling.MirroredDirectory("diana")
  assert isinstance(d, mirrored_directory.MirroredDirectory)
  assert d.RemoteExists()
  assert d.LocalExists()


@pytest.mark.diana
@pytest.mark.florence
@pytest.mark.parametrize("dir", ("music", "movies", "tv"))
def test_Ryangosling_mirrored_directory_exists(
  ryangosling: machine.Machine, dir: str
):
  """Test that mirrored directory exists."""
  d = ryangosling.MirroredDirectory(dir)
  assert isinstance(d, mirrored_directory.MirroredDirectory)


@pytest.mark.diana
@pytest.mark.florence
@pytest.mark.parametrize("dir", ("music", "movies", "tv"))
def test_Ryangosling_mirrored_directory_remote_exists(
  ryangosling: machine.Machine, dir: str
):
  """Test that mirrored directory remote exists."""
  d = ryangosling.MirroredDirectory(dir)
  assert d.RemoteExists()
  assert d.LocalExists()


@pytest.mark.diana
@pytest.mark.florence
@pytest.mark.parametrize("dir", ("music", "movies", "tv"))
def test_Ryangosling_mirrored_directory_local_exists(
  ryangosling: machine.Machine, dir: str
):
  """Test that mirrored directory local exists."""
  d = ryangosling.MirroredDirectory(dir)
  assert d.LocalExists()


if __name__ == "__main__":
  test.Main()
