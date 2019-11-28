"""Unit tests for //configure."""
import os
import pathlib
import sys
import tempfile

import pytest

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import test

FLAGS = app.FLAGS

# The path of the configure script.
CONFIG_SCRIPT = bazelutil.DataPath("phd/configure")

MODULE_UNDER_TEST = None  # No coverage.


@test.Fixture(scope="module")
def configure():
  """A test fixture which yields the configure script as an imported module."""
  with tempfile.TemporaryDirectory() as d:
    temp_module = pathlib.Path(d)
    with open(temp_module / "__init__.py", "w") as f:
      f.write("")
    with open(CONFIG_SCRIPT) as f:
      src = f.read()
    with open(temp_module / "configure.py", "w") as f:
      f.write(src)
    sys.path.append(d)
    import configure

    yield configure


def test_Mkdir(configure):
  """Test making a directory."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "a" / "b" / "c"
    configure.Mkdir(os.path.join(d, "a", "b", "c"))
    assert path.is_dir()
    # Test that you can call Mkdir() with an already existing directory.
    configure.Mkdir(os.path.join(d, "a", "b", "c"))
    assert path.is_dir()


if __name__ == "__main__":
  test.Main()
