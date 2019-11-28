"""Unit tests for //tools/code_style/linters:linters_lib."""
import os
import pathlib

import pytest

from labm8.py import app
from labm8.py import test
from tools.code_style.linters import linters_lib

FLAGS = app.FLAGS


def test_WhichOrDie_file_exists(tempdir: pathlib.Path):
  """Test that file can be found in PATH."""
  os.environ["PATH"] = str(tempdir)
  (tempdir / "a").touch()
  assert linters_lib.WhichOrDie("a")


def test_WhichOrDie_file_doesnt_exist(tempdir: pathlib.Path):
  """Test that error raised when file not in PATH."""
  os.environ["PATH"] = str(tempdir)
  with test.Raises(SystemExit):
    linters_lib.WhichOrDie("a")


if __name__ == "__main__":
  test.Main()
