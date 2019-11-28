"""Unit tests for //util/photolib/shutterbug/shutterbug.py."""
import pathlib
import tempfile

import pytest

from labm8.py import app
from labm8.py import test
from util.photolib.shutterbug import shutterbug

FLAGS = app.FLAGS

# Test fixtures.


@pytest.fixture(scope="function")
def photodir(tempdir: pathlib.Path) -> pathlib.Path:
  """Test fixture to create a """
  with open(tempdir / "a.jpg", "w") as f:
    f.write("Image A")
  with open(tempdir / "b.jpg", "w") as f:
    f.write("Image B")
  with open(tempdir / "c.jpg", "w") as f:
    f.write("Image C")
  # Create fake DS_Store files.
  (tempdir / ".DS_Store").touch()
  (tempdir / "._DS_Store").touch()
  yield tempdir


# Utility functions.


def _AssertIsPhotoDir(path: pathlib.Path) -> None:
  """Check that photodir (as generated by photodir test fixture) is valid."""
  assert (path / "a.jpg").is_file()
  assert (path / "b.jpg").is_file()
  assert (path / "c.jpg").is_file()
  with open(path / "a.jpg") as f:
    assert "Image A" == f.read()
  with open(path / "b.jpg") as f:
    assert "Image B" == f.read()
  with open(path / "c.jpg") as f:
    assert "Image C" == f.read()


# PathTuplesToChunk() tests.


def test_PathTuplesToChunk_directory_not_found(
  tempdir: pathlib.Path, photodir: pathlib.Path
):
  """A ValueError is raised if any of the source directories do not exist."""
  with pytest.raises(ValueError) as e_ctx:
    shutterbug.PathTuplesToChunk([photodir, tempdir / "foo"])
  assert str(e_ctx.value) == f"{tempdir}/foo not found"


def test_PathTuplesToChunk_photodir(photodir: pathlib.Path):
  """Test return value of path tuples."""
  files = sorted(shutterbug.PathTuplesToChunk([photodir]))
  print("FILES", files)
  assert files == [
    (str(photodir / "a.jpg"), str(photodir)),
    (str(photodir / "b.jpg"), str(photodir)),
    (str(photodir / "c.jpg"), str(photodir)),
  ]


# Integration tests.


def test_end_to_end(tempdir: pathlib.Path, photodir: pathlib.Path):
  """Test end to end packing and unpacking a photo directory."""
  _AssertIsPhotoDir(photodir)
  _ = shutterbug
  shutterbug.MakeChunks([photodir], tempdir, int(1e6))
  assert (tempdir / "chunk_001").is_dir()
  assert (tempdir / "chunk_001" / "README.txt").is_file()
  assert (tempdir / "chunk_001" / "MANIFEST.txt").is_file()
  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    out_dir = pathlib.Path(d)
    shutterbug.unchunk(tempdir, out_dir)
    _AssertIsPhotoDir(out_dir)


if __name__ == "__main__":
  test.Main()
