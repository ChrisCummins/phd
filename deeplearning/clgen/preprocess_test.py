"""Unit tests for //deeplearning/clgen:sample_observers."""
import pathlib

import pytest

from deeplearning.clgen import preprocess
from labm8.py import app
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def contentfiles(tempdir: pathlib.Path) -> pathlib.Path:
  fs.Write(tempdir / "a.txt", "int main() {}".encode("utf-8"))
  fs.Write(tempdir / "b.txt", "invalid syntax".encode("utf-8"))
  yield tempdir


def test_Preprocess(contentfiles: pathlib.Path, tempdir2: pathlib.Path):
  preprocess.Preprocess(
    contentfiles,
    tempdir2,
    [
      "deeplearning.clgen.preprocessors.cxx:Compile",
      "deeplearning.clgen.preprocessors.cxx:ClangFormat",
    ],
  )
  assert (tempdir2 / "a.txt").is_file()
  assert not (tempdir2 / "b.txt").is_file()

  with open(tempdir2 / "a.txt") as f:
    assert (
      f.read()
      == """int main() {
}"""
    )


if __name__ == "__main__":
  test.Main()
