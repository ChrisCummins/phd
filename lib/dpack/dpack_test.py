"""Unit tests for //lib/dpack:dpack."""
import sys
import tempfile

import pathlib
import pytest
from absl import app

from lib.dpack import dpack
from lib.dpack.proto import dpack_pb2

# The sha256sum of an empty file.
SHA256_EMPTY_FILE = ('e3b0c44298fc1c149afbf4c8996fb924'
                     '27ae41e4649b934ca495991b7852b855')


def test_IsPackage_files():
  # Test that _IsPackage() accepts files with '.dpack.tar.bz2' extension).
  with tempfile.NamedTemporaryFile() as f:
    assert not dpack._IsPackage(pathlib.Path(f.name))
  with tempfile.NamedTemporaryFile(suffix='.txt') as f:
    assert not dpack._IsPackage(pathlib.Path(f.name))
  with tempfile.NamedTemporaryFile(suffix='.tar.bz2') as f:
    assert not dpack._IsPackage(pathlib.Path(f.name))
  with tempfile.NamedTemporaryFile(suffix='.dpack.tar.bz2') as f:
    assert dpack._IsPackage(pathlib.Path(f.name))


def test_IsPackage_directory():
  # Test that _IsPackage() accepts a directory.
  with tempfile.TemporaryDirectory() as d:
    assert dpack._IsPackage(pathlib.Path(d))


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


def test_SetDataPackageFileAttributes_empty_file():
  # Test that file attributes are set.
  df = dpack_pb2.DataPackageFile()
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    dpack.SetDataPackageFileAttributes(pathlib.Path(d), 'a', df)

  assert df.relative_path == 'a'
  assert not df.size_in_bytes
  assert df.checksum_hash == dpack_pb2.SHA256
  assert df.checksum == SHA256_EMPTY_FILE


def test_DataPackageFileAttributesAreValid_missing_file():
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  with tempfile.TemporaryDirectory() as d:
    assert not dpack.DataPackageFileAttributesAreValid(pathlib.Path(d), df)


def test_DataPackageFileAttributesAreValid_unknown_checksum_hash():
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    assert not dpack.DataPackageFileAttributesAreValid(pathlib.Path(d), df)


def test_DataPackageFileAttributesAreValid_different_checksum():
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  df.checksum_hash = dpack_pb2.SHA256
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    assert not dpack.DataPackageFileAttributesAreValid(pathlib.Path(d), df)


def test_DataPackageFileAttributesAreValid_different_size():
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  df.checksum_hash = dpack_pb2.SHA256
  df.checksum = SHA256_EMPTY_FILE
  df.size_in_bytes = 10  # An empty file has size 0
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    assert not dpack.DataPackageFileAttributesAreValid(pathlib.Path(d), df)


def test_DataPackageFileAttributesAreValid_match():
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  df.checksum_hash = dpack_pb2.SHA256
  df.checksum = SHA256_EMPTY_FILE
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    assert dpack.DataPackageFileAttributesAreValid(pathlib.Path(d), df)


def test_MergeManifests_comments():
  # Test that the comments from the old manifest are copied to the new.
  d1 = dpack_pb2.DataPackage()
  f1 = d1.file.add()
  f1.relative_path = 'a'
  d2 = dpack_pb2.DataPackage()
  d2.comment = 'abc'
  f2 = d2.file.add()
  f2.comment = 'def'
  f2.relative_path = 'a'
  dpack.MergeManifests(d1, d2)
  assert d1.comment == d2.comment
  assert d1.file[0].comment == d2.file[0].comment


def test_MergeManifests_file_attributes():
  # Test that file attributes are not merged.
  d1 = dpack_pb2.DataPackage()
  f1 = d1.file.add()
  f1.relative_path = 'a'
  f1.size_in_bytes = 1
  f1.checksum_hash = dpack_pb2.SHA1
  f1.checksum = 'abc'
  d2 = dpack_pb2.DataPackage()
  f2 = d2.file.add()
  f2.relative_path = 'a'
  f2.size_in_bytes = 2
  f2.checksum_hash = dpack_pb2.MD5
  f2.checksum = 'def'
  dpack.MergeManifests(d1, d2)
  assert d1.file[0].size_in_bytes == 1
  assert d1.file[0].checksum_hash == dpack_pb2.SHA1
  assert d1.file[0].checksum == 'abc'


def test_MergeManifests_missing_files():
  # Test that files that only appear in one manifest are not modified.
  d1 = dpack_pb2.DataPackage()
  f1 = d1.file.add()
  f1.relative_path = 'a'
  f1.comment = 'abc'
  d2 = dpack_pb2.DataPackage()
  f2 = d2.file.add()
  f2.relative_path = 'b'
  f2.comment = 'def'
  dpack.MergeManifests(d1, d2)
  assert d1.file[0].comment == 'abc'
  assert d2.file[0].comment == 'def'


def test_CreatePackageManifest_empty_directory():
  # Test the manifest of an empty directory.
  with tempfile.TemporaryDirectory() as d:
    m = dpack.CreatePackageManifest(pathlib.Path(d), [])
  assert m.comment == ''
  assert m.utc_epoch_ms_packaged
  assert not len(m.file)


def test_CreatePackageManifest():
  # Test the manifest of an empty directory.
  with tempfile.TemporaryDirectory() as d:
    (pathlib.Path(d) / 'a').touch()
    m = dpack.CreatePackageManifest(pathlib.Path(d), [pathlib.Path('a')])
  assert len(m.file) == 1
  assert m.file[0].comment == ''
  assert not m.file[0].size_in_bytes
  assert m.file[0].checksum_hash == dpack_pb2.SHA256
  assert m.file[0].checksum == SHA256_EMPTY_FILE


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
