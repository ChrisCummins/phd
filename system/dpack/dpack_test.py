"""Unit tests for //lib/dpack:dpack."""
import pathlib
import tempfile

from labm8 import app
from labm8 import test
from system.dpack import dpack
from system.dpack.proto import dpack_pb2

FLAGS = app.FLAGS

# The sha256sum of an empty file.
SHA256_EMPTY_FILE = (
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')


def test_IsPackage_files():
  """Test that _IsPackage() accepts files with '.dpack.tar.bz2' extension."""
  with tempfile.NamedTemporaryFile() as f:
    assert not dpack._IsPackage(pathlib.Path(f.name))
  with tempfile.NamedTemporaryFile(suffix='.txt') as f:
    assert not dpack._IsPackage(pathlib.Path(f.name))
  with tempfile.NamedTemporaryFile(suffix='.tar.bz2') as f:
    assert not dpack._IsPackage(pathlib.Path(f.name))
  with tempfile.NamedTemporaryFile(suffix='.dpack.tar.bz2') as f:
    assert dpack._IsPackage(pathlib.Path(f.name))


def test_IsPackage_directory(tempdir: pathlib.Path):
  """Test that _IsPackage() accepts a directory."""
  assert dpack._IsPackage(tempdir)


def test_GetFilesInDirectory_empty_dir(tempdir: pathlib.Path):
  """Test that an empty directory has no contents."""
  assert not dpack.GetFilesInDirectory(tempdir, [])


def test_GetFilesInDirectory_leaf_files(tempdir: pathlib.Path):
  """Test that files in the directory are returned."""
  # Start with one file.
  (tempdir / 'a').touch()
  assert set(dpack.GetFilesInDirectory(tempdir, [])) == {pathlib.Path('a')}
  # Add a second file.
  (tempdir / 'b').touch()
  assert set(dpack.GetFilesInDirectory(tempdir, [])) == {
      pathlib.Path('a'),
      pathlib.Path('b'),
  }
  # Add a third file.
  (tempdir / 'c').touch()
  assert set(dpack.GetFilesInDirectory(tempdir, [])) == {
      pathlib.Path('a'),
      pathlib.Path('b'),
      pathlib.Path('c'),
  }


def test_GetFilesInDirectory_subdir_relpath(tempdir: pathlib.Path):
  """Test that relative paths to files in a subdirectory are returned."""
  # Create files: [ sub/a, sub/sub/b ]
  (tempdir / 'sub').mkdir()
  (tempdir / 'sub' / 'a').touch()
  (tempdir / 'sub' / 'sub').mkdir()
  (tempdir / 'sub' / 'sub' / 'b').touch()
  assert set(dpack.GetFilesInDirectory(
      tempdir, [])) == {pathlib.Path('sub/a'),
                        pathlib.Path('sub/sub/b')}


def test_GetFilesInDirectory_exclude_by_name(tempdir: pathlib.Path):
  """Test that filenames which exactly match an exclude pattern are excluded."""
  # Create files: [ a, foo, sub/foo ]
  (tempdir / 'a').touch()
  (tempdir / 'foo').touch()
  (tempdir / 'sub').mkdir()
  (tempdir / 'sub' / 'foo').touch()
  # Exclude pattern 'foo' does not exclude subdir 'foo'.
  assert set(dpack.GetFilesInDirectory(
      tempdir, ['foo'])) == {pathlib.Path('a'),
                             pathlib.Path('sub/foo')}


def test_GetFilesInDirectory_exclude_subdir(tempdir: pathlib.Path):
  """Test that files in subdirs can be excluded."""
  # Create files: [ a, foo, sub/foo ]
  (tempdir / 'a').touch()
  (tempdir / 'foo').touch()
  (tempdir / 'sub').mkdir()
  (tempdir / 'sub' / 'foo').touch()
  (tempdir / 'sub' / 'sub').mkdir()
  (tempdir / 'sub' / 'sub' / 'foo').touch()
  assert set(dpack.GetFilesInDirectory(tempdir, ['sub/foo'])) == {
      pathlib.Path('a'),
      pathlib.Path('foo'),
      pathlib.Path('sub/sub/foo')
  }
  assert set(dpack.GetFilesInDirectory(
      tempdir, ['*/foo'])) == {pathlib.Path('a'),
                               pathlib.Path('foo')}
  assert set(dpack.GetFilesInDirectory(
      tempdir, ['*/foo*'])) == {pathlib.Path('a'),
                                pathlib.Path('foo')}


def test_SetDataPackageFileAttributes_empty_file(tempdir: pathlib.Path):
  """Test that file attributes are set."""
  df = dpack_pb2.DataPackageFile()
  (tempdir / 'a').touch()
  dpack.SetDataPackageFileAttributes(tempdir, 'a', df)

  assert df.relative_path == 'a'
  assert not df.size_in_bytes
  assert df.checksum_hash == dpack_pb2.SHA256
  assert df.checksum == SHA256_EMPTY_FILE


def test_DataPackageFileAttributesAreValid_missing_file(tempdir: pathlib.Path):
  """If a file does not exist, attributes are not valid."""
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  assert not dpack.DataPackageFileAttributesAreValid(tempdir, df)


def test_DataPackageFileAttributesAreValid_unknown_checksum_hash(
    tempdir: pathlib.Path):
  """If no checksum hash is declared, attributes are not valid."""
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  (tempdir / 'a').touch()
  assert not dpack.DataPackageFileAttributesAreValid(tempdir, df)


def test_DataPackageFileAttributesAreValid_different_checksum(
    tempdir: pathlib.Path):
  """If checksum of file differs, attributes are not valid."""
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  df.checksum_hash = dpack_pb2.SHA256
  (tempdir / 'a').touch()
  assert not dpack.DataPackageFileAttributesAreValid(tempdir, df)


def test_DataPackageFileAttributesAreValid_different_size(
    tempdir: pathlib.Path):
  """If the size of a file is incorect, attributes are not valid."""
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  df.checksum_hash = dpack_pb2.SHA256
  df.checksum = SHA256_EMPTY_FILE
  df.size_in_bytes = 10  # An empty file has size 0
  (tempdir / 'a').touch()
  assert not dpack.DataPackageFileAttributesAreValid(tempdir, df)


def test_DataPackageFileAttributesAreValid_match(tempdir: pathlib.Path):
  """Test that file attributes can be correct."""
  df = dpack_pb2.DataPackageFile()
  df.relative_path = 'a'
  df.checksum_hash = dpack_pb2.SHA256
  df.checksum = SHA256_EMPTY_FILE
  (tempdir / 'a').touch()
  assert dpack.DataPackageFileAttributesAreValid(tempdir, df)


def test_MergeManifests_comments():
  """Test that comments from old manifests are copied to the new ones."""
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
  """Test that file attributes are not merged."""
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
  """Test that files that only appear in one manifest are not modified."""
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


def test_CreatePackageManifest_empty_directory(tempdir: pathlib.Path):
  """Test the manifest of an empty directory."""
  m = dpack.CreatePackageManifest(tempdir, [])
  assert m.comment == ''
  assert m.utc_epoch_ms_packaged
  assert not len(m.file)


def test_CreatePackageManifest(tempdir: pathlib.Path):
  """Test the manifest of an empty directory."""
  (tempdir / 'a').touch()
  m = dpack.CreatePackageManifest(tempdir, [pathlib.Path('a')])
  assert len(m.file) == 1
  assert m.file[0].comment == ''
  assert not m.file[0].size_in_bytes
  assert m.file[0].checksum_hash == dpack_pb2.SHA256
  assert m.file[0].checksum == SHA256_EMPTY_FILE


if __name__ == '__main__':
  test.Main()
