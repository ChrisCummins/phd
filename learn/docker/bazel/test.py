"""A test python file."""
import sys

import pytest

from labm8.py import app
from labm8.py import bazelutil

FLAGS = app.FLAGS


def test_datafile_read():
  """Test reading a data file."""
  with open(bazelutil.DataPath("phd/learn/docker/bazel/datafile.txt")) as f:
    assert f.read() == "Hello, Docker!\n"


def main(argv):
  """Main entry point."""
  app.Log(1, "Platform: %s", sys.platform)
  app.Log(1, "Exec:     %s", sys.executable)
  app.Log(1, "Args:     %s", " ".join(argv))
  if len(argv) > 1:
    app.Warning("Unknown arguments: '%s'", " ".join(argv[1:]))
  sys.exit(pytest.main([__file__, "-vv"]))


if __name__ == "__main__":
  app.FLAGS(["argv[0]", "-v=1"])
  app.RunWithArgs(main)
