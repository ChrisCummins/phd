"""Black box tests for //system/machines/ryangosling:ryangosling.pbtxt.

These tests will NOT pass if run on one of my personal machines.
"""
import pathlib
import sys

from labm8.py import app
from system.machines import machine

FLAGS = app.FLAGS

# The path of the installed config file.
_RYANGOSLING_MACHINE_CONFIG = pathlib.Path(
  "~/.local/var/machines/ryangosling.pbtxt"
).expanduser()


def Test(name, callback):
  value = callback()
  print(f"{name:20s} {'PASS' if value else 'FAIL'}")
  if not value:
    print("Test failured!", file=sys.stderr)
    sys.exit(1)


def Main():
  """Test that mirrored directories exists."""
  ryangosling = machine.Machine.FromFile(_RYANGOSLING_MACHINE_CONFIG)

  for mirrored_directory in ryangosling.mirrored_directories:
    print(
      "==========================================================================================================="
    )
    print(mirrored_directory)
    print(
      "==========================================================================================================="
    )
    Test("Local exists ...", mirrored_directory.LocalExists)
    Test("Remote exists ...", mirrored_directory.RemoteExists)
    print("")


if __name__ == "__main__":
  app.Run(Main)
