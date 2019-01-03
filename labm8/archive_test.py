"""Unit tests for //labm8:archive."""
import pathlib
import re
import zipfile

import pytest
from absl import flags

from labm8 import archive
from labm8 import test


FLAGS = flags.FLAGS


def Touch(path: pathlib.Path) -> pathlib.Path:
  with open(path, 'w') as f:
    pass
  return path


def test_Archive_path_not_found(tempdir: pathlib.Path):
  """Test that FileNotFound raised if path doesn't exist."""
  with pytest.raises(FileNotFoundError) as e_ctx:
    archive.Archive(tempdir / 'a.zip')
  assert str(e_ctx.value).startswith("No such file: '")


def test_Archive_no_suffix(tempdir: pathlib.Path):
  """Test that error raised if path has no suffix."""
  Touch(tempdir / 'a')
  with pytest.raises(archive.UnsupportedArchiveFormat) as e_ctx:
    archive.Archive(tempdir / 'a')
  assert str(e_ctx.value) == "Archive 'a' has no extension"


def test_Archive_assume_filename_no_suffix(tempdir: pathlib.Path):
  """Test that error raised if assumed path has no suffix."""
  Touch(tempdir / 'a.zip')
  with pytest.raises(archive.UnsupportedArchiveFormat) as e_ctx:
    archive.Archive(tempdir / 'a.zip', assume_filename='a')
  assert str(e_ctx.value) == "Archive 'a' has no extension"


@pytest.mark.parametrize('suffix', ('.foo', '.tar.abc'))
def test_Archive_unsupported_suffixes(tempdir: pathlib.Path, suffix: str):
  path = tempdir / f'a{suffix}'
  Touch(path)

  with pytest.raises(archive.UnsupportedArchiveFormat) as e_ctx:
    archive.Archive(path)
  assert re.match(f"Unsupported file extension '(.+)' for archive 'a{suffix}'",
                  str(e_ctx.value))


def test_Archive_single_file_zip(tempdir: pathlib.Path):
  """Test context manager for a single file zip."""
  # Create an archive with a single file.
  path = tempdir / 'a.zip'
  with zipfile.ZipFile(path, "w") as a:
    with a.open('a.txt', 'w') as f:
      f.write("Hello, world!".encode("utf-8"))

  # Open the archive and check the contents.
  with archive.Archive(path) as d:
    assert (d / 'a.txt').is_file()
    assert len(list(d.iterdir())) == 1
    with open(d / 'a.txt') as f:
      assert f.read() == "Hello, world!"


def test_Archive_single_file_zip_ExtractAll(tempdir: pathlib.Path):
  """Test ExtractAll for a single file zip."""
  # Create an archive with a single file.
  path = tempdir / 'a.zip'
  with zipfile.ZipFile(path, "w") as a:
    with a.open('a.txt', 'w') as f:
      f.write("Hello, world!".encode("utf-8"))

  # Open the archive and check that it still exists.
  archive.Archive(path).ExtractAll(tempdir)
  assert (tempdir / 'a.zip').is_file()

  # Check the archive contents.
  assert (tempdir / 'a.txt').is_file()
  assert len(list(tempdir.iterdir())) == 2  # the zip file and a.txt
  with open(tempdir / 'a.txt') as f:
    assert f.read() == "Hello, world!"


def test_Archive_single_file_zip_ExtractAll_parents(tempdir: pathlib.Path):
  """Test that ExtractAll creates necessary parent directories"""
  # Create an archive with a single file.
  path = tempdir / 'a.zip'
  with zipfile.ZipFile(path, "w") as a:
    with a.open('a.txt', 'w') as f:
      f.write("Hello, world!".encode("utf-8"))

  # Open the archive and check that it still exists.
  archive.Archive(path).ExtractAll(tempdir / 'foo/bar/car')
  assert (tempdir / 'a.zip').is_file()

  # Check the archive contents.
  assert (tempdir / 'foo/bar/car/a.txt').is_file()
  assert len(list(tempdir.iterdir())) == 2  # the zip file and 'foo/'
  with open(tempdir / 'foo/bar/car/a.txt') as f:
    assert f.read() == "Hello, world!"


if __name__ == '__main__':
  test.Main()
