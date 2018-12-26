"""Unit tests for //labm8:archive."""
import pathlib
import re
import sys
import tempfile
import typing
import zipfile

import pytest
from absl import app
from absl import flags

from labm8 import archive


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield pathlib.Path(d)


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


@pytest.mark.parametrize('suffix', ('.zip',))
def test_Archive_single_file_zip(tempdir: pathlib.Path, suffix: str):
  """Short summary of test."""
  # Create an archive with a single file.
  path = tempdir / f'a{suffix}'
  with zipfile.ZipFile(path, "w") as a:
    with a.open('a.txt', 'w') as f:
      f.write("Hello, world!".encode("utf-8"))

  # Open the archive and check the contents.
  with archive.Archive(path) as d:
    assert (d / 'a.txt').is_file()
    assert len(list(d.iterdir())) == 1
    with open(d / 'a.txt') as f:
      assert f.read() == "Hello, world!"


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
