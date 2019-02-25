"""Dpack creates structured packages of data files.

Dpack creates an archive of a directory with a manifest. The manifest describes
each of the files within the directory. Use dpack to create consistent and well
documented collections of data files.
"""
import fnmatch
import os
import pathlib
import sys
import tarfile
import typing

from absl import app
from absl import flags
from absl import logging

from labm8 import crypto
from labm8 import fs
from labm8 import labdate
from labm8 import pbutil
from system.dpack.proto import dpack_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('package', None, 'The path of the target package.')
flags.DEFINE_string('sidecar', None, 'The path of the archive sidecar.')
flags.DEFINE_list(
    'exclude', [], 'A list of patterns to exclude from the package. Supports '
    'UNIX-style globbing: *,?,[],[!].')
flags.DEFINE_bool(
    'init', False, 'If set, create the package MANIFEST.pbtxt file. This will '
    'not overwrite an existing manifest file.')
flags.DEFINE_bool(
    'update', False,
    'If set, update the file attributes in MANIFEST.pbtxt file. '
    'This can only be used in conjunction with --init flag. '
    'If no MANIFEST.pbtxt exists, it is created.')
flags.DEFINE_bool('pack', False, 'If set, create the package archive.')


def _IsPackage(path: pathlib.Path) -> bool:
  """Check that a path is a package: either a .dpack.tar.bz2 file or a dir."""
  if path.is_dir():
    return True
  else:
    return path.is_file() and path.suffixes == ['.dpack', '.tar', '.bz2']


# The --package argument points to either a directory or an archive file.
flags.register_validator(
    'package',
    # Flags validation occurs whenever this file is imported. During unit
    # testing we have no value for this flag, so the validator should only
    # run if the flag is present.
    lambda path: pathlib.Path(path).exists() if path else True,
    message='--package path not found.')


def _IsManifest(path: pathlib.Path) -> bool:
  """Check if a path contains a DataPackafe file."""
  return pbutil.ProtoIsReadable(path, dpack_pb2.DataPackage())


# The --sidecar argument optionally points to a DataPackage message.
flags.register_validator(
    'sidecar',
    lambda path: _IsManifest(path) if path else True,
    message='--sidecar path not found.')

# A list of filename patterns to exclude from all data packages.
ALWAYS_EXCLUDE_PATTERNS = [
    'MANIFEST.pbtxt',  # No self-reference.
    '.DS_Store',
    '._.DS_Store',
    '*/.DS_Store',
    '*/._.DS_Store',
    '.com.apple.timemachine.supported',
    '*/.com.apple.timemachine.supported',
    '.sync.ffs_db',
    '*/.sync.ffs_db',
]


def GetFilesInDirectory(
    directory: pathlib.Path,
    exclude_patterns: typing.List[str]) -> typing.List[pathlib.Path]:
  """Recursively list all files in a directory.

  Returns relative paths of all files in a directory which do not match the
  exclude patterns. The list of exclude patterns supports UNIX style globbing.

  Args:
    directory: The path to the directory.
    exclude_patterns: A list of patterns to exclude.

  Returns:
    A list of paths.
  """
  exclude_patterns = set(exclude_patterns + ALWAYS_EXCLUDE_PATTERNS)
  files = []
  for path in sorted(fs.lsfiles(directory, recursive=True)):
    for pattern in exclude_patterns:
      if fnmatch.fnmatch(path, pattern):
        logging.info('- %s', path)
        break
    else:
      logging.info('+ %s', path)
      files.append(pathlib.Path(path))
  return files


def SetDataPackageFileAttributes(package_root: pathlib.Path,
                                 relpath: pathlib.Path,
                                 f: dpack_pb2.DataPackageFile) -> None:
  """Set the file attributes of a DataPackageFile message.

  Args:
    package_root: The root of the package.
    relpath: The path to the file, relative to package_root.
    f: A DataPackageFile instance.
  """
  abspath = package_root / relpath
  f.relative_path = str(relpath)
  f.size_in_bytes = abspath.stat().st_size
  f.checksum = crypto.sha256_file(abspath)
  f.checksum_hash = dpack_pb2.SHA256


def DataPackageFileAttributesAreValid(package_root: pathlib.Path,
                                      f: dpack_pb2.DataPackageFile) -> bool:
  """Check that the values in a DataPackageFile match the real file.

  Args:
    package_root: The root of the package.
    f: A DataPackageFile instance.

  Returns:
    True if the DataPackageFile fields match what is found in the filesystem.
  """
  abspath = package_root / f.relative_path
  if not abspath.is_file():
    logging.warning("'%s' has vanished", f.relative_path)
    return False

  size_in_bytes = abspath.stat().st_size
  if f.size_in_bytes != size_in_bytes:
    logging.warning("the contents of '%s' has changed", f.relative_path)
    return False

  hash_fn = dpack_pb2.ChecksumHash.Name(f.checksum_hash).lower()
  try:
    checksum_fn = getattr(crypto, hash_fn + '_file')
  except AttributeError:
    logging.warning("unknown value for field checksum_hash in '%s'",
                    f.relative_path)
    return False

  checksum = checksum_fn(abspath)
  if f.checksum != checksum:
    logging.warning(
        "the contents of '%s' have changed but the size remains "
        "the same", f.relative_path)
    return False

  return True


def MergeManifests(new: dpack_pb2.DataPackage,
                   old: dpack_pb2.DataPackage) -> None:
  """Transfer non-file attribute fields from old to new manifests.

  This copies over the comment and package date fields from the old manifest
  to the new manifest. File attributes are not updated.

  Args:
    new: The manifest to merge to.
    old: The manifest to merge fields from.
  """
  new.comment = old.comment
  new.utc_epoch_ms_packaged = old.utc_epoch_ms_packaged
  old_files = {f.relative_path: f for f in old.file}
  for f in new.file:
    if f.relative_path in old_files:
      f.comment = old_files[f.relative_path].comment


def CreatePackageManifest(
    package_root: pathlib.Path,
    contents: typing.List[pathlib.Path]) -> dpack_pb2.DataPackage:
  """Create a DataPackage message for the contents of a package.

  Args:
    package_root: The root of the package.
    contents: A list of relative paths to files to include.

  Returns:
    A DataPackage instance with attributes set.
  """
  manifest = dpack_pb2.DataPackage()
  manifest.comment = ''
  manifest.utc_epoch_ms_packaged = labdate.MillisecondsTimestamp(
      labdate.GetUtcMillisecondsNow())
  for path in contents:
    f = manifest.file.add()
    SetDataPackageFileAttributes(package_root, path, f)
    f.comment = f.comment or ''
  return manifest


def PackageManifestIsValid(package_root: pathlib.Path,
                           manifest: dpack_pb2.DataPackage) -> bool:
  """Check that the package manifest is correct.

  Args:
    package_root: The root of the package.
    manifest: A DataPackage instance describing the package.

  Returns:
    True if the manifest matches the contents of the file system, else False.
  """
  return all(
      DataPackageFileAttributesAreValid(package_root, f) for f in manifest.file)


def CreatePackageArchive(package_dir: pathlib.Path,
                         manifest: dpack_pb2.DataPackage,
                         archive_path: pathlib.Path) -> None:
  """Create a tarball of the package.

  Args:
    package_dir: The root of the package.
    manifest: A DataPackage manifest instance.
    archive_path: The path of the archive to create.

  Raises:
    OSError: If archive_path already exists.
  """
  if archive_path.exists():
    raise OSError(f'Refusing to overwrite {archive_path}.')

  # Change to the package directory so that relative paths within the archive
  # are preserved.
  os.chdir(package_dir.parent)
  with tarfile.open(archive_path.absolute(), 'w:bz2') as tar:
    path = os.path.join(package_dir.name, 'MANIFEST.pbtxt')
    logging.info('+ %s', path)
    tar.add(path)
    for f in manifest.file:
      logging.info('+ %s', f.relative_path)
      path = os.path.join(package_dir.name, f.relative_path)
      tar.add(path)
  logging.info('Created %s', archive_path.absolute())


def CreatePackageArchiveSidecar(archive_path: pathlib.Path,
                                manifest: dpack_pb2.DataPackage,
                                sidecar_path: pathlib.Path) -> None:
  """Create a sidecar manifest to accompany an archive.

  Args:
    archive_path: The path of the archive tarball.
    manifest: A DataPackage manifest instance.
    sidecar_path: The path of the sidecar to create

  Raises:
    OSError: If sidecar_path already exists, or archive_path does not.
  """
  if sidecar_path.exists():
    raise OSError(f'Refusing to overwrite {sidecar_path}.')
  if not archive_path.is_file():
    raise OSError(f'Archive {archive_path} does not exist')

  sidecar = dpack_pb2.DataPackage()
  sidecar.CopyFrom(manifest)
  # Clear the file attributes. Only the file names and comments are stored in the sidecar.
  for f in sidecar.file:
    if not f.comment:
      f.ClearField("comment")
    f.ClearField("size_in_bytes")
    f.ClearField("checksum_hash")
    f.ClearField("checksum")
  sidecar.checksum_hash = dpack_pb2.SHA256
  sidecar.checksum = crypto.sha256_file(archive_path)
  pbutil.ToFile(sidecar, sidecar_path)
  logging.info('Wrote %s', sidecar_path.absolute())


def PackDataPackage(package_dir: pathlib.Path) -> None:
  """Create an archive and sidecar of a package."""
  manifest = pbutil.FromFile(package_dir / 'MANIFEST.pbtxt',
                             dpack_pb2.DataPackage())
  PackageManifestIsValid(package_dir, manifest)
  archive_path = (
      package_dir / f'../{package_dir.name}.dpack.tar.bz2').resolve()
  sidecar_path = (package_dir / f'../{package_dir.name}.dpack.pbtxt').resolve()
  CreatePackageArchive(package_dir, manifest, archive_path)
  CreatePackageArchiveSidecar(archive_path, manifest, sidecar_path)


def InitManifest(package_dir: pathlib.Path, contents: typing.List[pathlib.Path],
                 update: bool) -> None:
  """Write the MANIFEST.pbtxt file for a package."""
  manifest = CreatePackageManifest(package_dir, contents)
  manifest_path = package_dir / 'MANIFEST.pbtxt'
  if update and pbutil.ProtoIsReadable(manifest_path, dpack_pb2.DataPackage()):
    old = pbutil.FromFile(manifest_path, dpack_pb2.DataPackage())
    MergeManifests(manifest, old)
  elif manifest_path.is_file():
    raise OSError('Refusing to overwrite MANIFEST.pbtxt file.')
  pbutil.ToFile(manifest, manifest_path)
  logging.info('Wrote %s', manifest_path.absolute())


def VerifyManifest(package_dir: pathlib.Path) -> bool:
  """Verify that the MANIFEST.pbtext file matches the contents."""
  if not (package_dir / 'MANIFEST.pbtxt').is_file():
    logging.info('%s/MANIFEST.pbtxt missing, nothing to do.', package_dir)
    return False
  manifest = pbutil.FromFile(package_dir / 'MANIFEST.pbtxt',
                             dpack_pb2.DataPackage())
  if not PackageManifestIsValid(package_dir, manifest):
    logging.error('Package %s contains errors.', package_dir)
    return False
  logging.info('%s verified. No changes to files in the manifest.', package_dir)
  return True


def SidecarIsValid(archive: pathlib.Path, sidecar: pathlib.Path) -> None:
  """Check the archive matches the attributes in the sidecar."""
  sidecar_manifest = pbutil.FromFile(sidecar, dpack_pb2.DataPackage())

  hash_fn = dpack_pb2.ChecksumHash.Name(sidecar_manifest.checksum_hash).lower()
  try:
    checksum_fn = getattr(crypto, hash_fn + '_file')
  except AttributeError:
    logging.warning("unknown value for field checksum_hash in manifest")
    return False
  checksum = checksum_fn(archive)
  if sidecar_manifest.checksum != checksum:
    logging.warning("the contents of '%s' have changed", archive.absolute())
    return False
  logging.info('Package verified using the sidecar.')
  return True


def main(argv) -> None:
  """Main entry point."""
  # Validate flags.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.package:
    raise app.UsageError('--package argument is required.')

  if FLAGS.update and not FLAGS.init:
    logging.warning('--update flag ignored.')

  package = pathlib.Path(FLAGS.package)

  # Perform the requested action.
  if FLAGS.pack:
    PackDataPackage(package)
  elif FLAGS.init:
    contents = GetFilesInDirectory(package, FLAGS.exclude)
    InitManifest(package, contents, update=FLAGS.update)
  elif FLAGS.sidecar:
    if not SidecarIsValid(package, pathlib.Path(FLAGS.sidecar)):
      sys.exit(1)
  else:
    if not VerifyManifest(package):
      sys.exit(1)


if __name__ == '__main__':
  app.run(main)
