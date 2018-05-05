"""Unit tests for //lib/dpack:dpack."""
import sys
import tempfile

import pathlib
import pytest
from absl import app

from lib.dpack import dpack


def test_GetFilesInDirectory_empty_dir():
  # Test that an empty directory has no contents.
  with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp)
    assert not dpack.GetFilesInDirectory(path, [])


def test_GetFilesInDirectory_leaf_files():
  # Test that files in the directory are returned.
  with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp)
    # Start with one file.
    (path / 'a').touch()
    assert set(dpack.GetFilesInDirectory(path, [])) == {
      pathlib.Path('a')
    }
    # Add a second file.
    (path / 'b').touch()
    assert set(dpack.GetFilesInDirectory(path, [])) == {
      pathlib.Path('a'), pathlib.Path('b'),
    }
    # Add a third file.
    (path / 'c').touch()
    assert set(dpack.GetFilesInDirectory(path, [])) == {
      pathlib.Path('a'), pathlib.Path('b'), pathlib.Path('c'),
    }


def test_GetFilesInDirectory_subdir_relpath():
  # Test that relative paths to files in a subdirectory are returned.
  with tempfile.TemporaryDirectory() as tmp:
    # Create files: [ sub/a, sub/sub/b ]
    path = pathlib.Path(tmp)
    (path / 'sub').mkdir()
    (path / 'sub' / 'a').touch()
    (path / 'sub' / 'sub').mkdir()
    (path / 'sub' / 'sub' / 'b').touch()
    assert set(dpack.GetFilesInDirectory(path, [])) == {
      pathlib.Path('sub/a'), pathlib.Path('sub/sub/b')
    }


def test_GetFilesInDirectory_exclude_by_name():
  # Test that filenames which exactly match an exclude pattern are excluded.
  with tempfile.TemporaryDirectory() as tmp:
    # Create files: [ a, foo, sub/foo ]
    path = pathlib.Path(tmp)
    (path / 'a').touch()
    (path / 'foo').touch()
    (path / 'sub').mkdir()
    (path / 'sub' / 'foo').touch()
    # Exclude pattern 'foo' does not exclude subdir 'foo'.
    assert set(dpack.GetFilesInDirectory(path, ['foo'])) == {
      pathlib.Path('a'), pathlib.Path('sub/foo')
    }


def test_GetFilesInDirectory_exclude_subdir():
  # Test that files in subdirs can be excluded.
  with tempfile.TemporaryDirectory() as tmp:
    # Create files: [ a, foo, sub/foo ]
    path = pathlib.Path(tmp)
    (path / 'a').touch()
    (path / 'foo').touch()
    (path / 'sub').mkdir()
    (path / 'sub' / 'foo').touch()
    (path / 'sub' / 'sub').mkdir()
    (path / 'sub' / 'sub' / 'foo').touch()
    assert set(dpack.GetFilesInDirectory(path, ['sub/foo'])) == {
      pathlib.Path('a'), pathlib.Path('foo'), pathlib.Path('sub/sub/foo')
    }
    assert set(dpack.GetFilesInDirectory(path, ['*/foo'])) == {
      pathlib.Path('a'), pathlib.Path('foo')
    }
    assert set(dpack.GetFilesInDirectory(path, ['*/foo*'])) == {
      pathlib.Path('a'), pathlib.Path('foo')
    }


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
