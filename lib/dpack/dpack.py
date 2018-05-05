"""Dpack creates structured packages of data files.

Dpack creates an archive of a directory with a manifest. The manifest describes
each of the files within the directory. Use dpack to create consistent and well
documented collections of data files.
"""
import fnmatch
import sys

import pathlib
import typing
from absl import app
from absl import flags
from absl import logging

from lib.dpack.proto import dpack_pb2
from lib.labm8 import crypto
from lib.labm8 import fs
from lib.labm8 import labdate
from lib.labm8 import pbutil

FLAGS = flags.FLAGS

flags.DEFINE_string('package', None,
                    'The path of the target package.')
flags.DEFINE_list('exclude', [],
                  'A list of patterns to exclude from the package. Supports '
                  'UNIX-style globbing: *,?,[],[!].')
flags.DEFINE_bool('init', False,
                  'If set, create the package MANIFEST.pbtxt file. This will '
                  'not overwrite an existing manifest file.')
flags.DEFINE_bool('update', False,
                  'If set, update the file attributes in MANIFEST.pbtxt file. '
                  'This can only be used in conjunction with --init flag. '
                  'If no MANIFEST.pbtxt exists, it is created.')
flags.DEFINE_bool('pack', False,
                  'If set, create the package archive.')

flags.register_validator(
    'package',
    # Flags validation occurs whenever this file is imported. During unit
    # testing we have no value for this flag, so the validator should only
    # run if the flag is present.
    lambda path: pathlib.Path(path).exists() if path else True,
    message='--package path not found.')

# A list of filename patterns to exclude from all data packages.
ALWAYS_EXCLUDE_PATTERNS = [
  'MANIFEST.pbtxt',  # No self-reference.
  '.DS_Store',
  '*/.DS_Store/*',
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
  """TODO.

  Args:
    package_root:
    relpath:
    f:

  Returns:

  """
  abspath = package_root / relpath
  f.relative_path = str(relpath)
  f.size_in_bytes = abspath.stat().st_size
  f.checksum = crypto.sha256_file(abspath)
  f.checksum_hash = dpack_pb2.SHA256


def VerifyDataPackageFileAttributes(package_root: pathlib.Path,
                                    f: dpack_pb2.DataPackageFile) -> bool:
  abspath = package_root / f.relative_path
  size_in_bytes = abspath.stat().st_size
  if f.size_in_bytes != size_in_bytes:
    logging.warning("the contents of '%s' has changed", f.relative_path)
    return False

  hash_fn = dpack_pb2.ChecksumHash.Name(f.checksum_hash).lower()
  crypto_fn = getattr(crypto, hash_fn + '_file')
  checksum = crypto_fn(abspath)
  if f.checksum != checksum:
    logging.warning("the contents of '%s' have changed but the size remains "
                    "the same", abspath)
    return False

  return True


def _MergeComments(new: dpack_pb2.DataPackage,
                   old: dpack_pb2.DataPackage):
  new.comment = old.comment
  old_files = {f.relative_path: f for f in old.file}
  for f in new.file:
    if f.relative_path in old_files:
      f.comment = old_files[f.relative_path].comment


def CreatePackageManifest(
    package_root: pathlib.Path,
    exclude_patterns: typing.List[str]) -> dpack_pb2.DataPackage:
  """TODO.

  Args:
    package_root:
    exclude_patterns:

  Returns:
    A DataPackage instance with attributes set.
  """
  manifest = dpack_pb2.DataPackage()
  manifest.comment = ''
  manifest.utc_epoch_ms_packaged = labdate.MillisecondsTimestamp(
      labdate.GetUtcMillisecondsNow())
  for path in GetFilesInDirectory(package_root, exclude_patterns):
    f = manifest.file.add()
    SetDataPackageFileAttributes(package_root, path, f)
    f.comment = f.comment or ''
  return manifest


def VerifyPackageManifest(package_root: pathlib.Path,
                          manifest: dpack_pb2.DataPackage) -> bool:
  """TODO.

  Args:
    package_root:
    manifest:

  Returns:

  """
  return all(VerifyDataPackageFileAttributes(package_root, f)
             for f in manifest.file)


def CreatePackageArchive(package_dir: pathlib.Path, manifest: dpack_pb2.DataPackage,
                         archive_path: pathlib.Path) -> None:
  """TODO.

  Args:
    package_dir:
    manifest:
    archive_path:

  Returns:

  Raises:
    OSError:
  """
  if archive_path.exists():
    raise OSError(f'Refusing to overwrite {archive_path}.')

  logging.info('Creating archive %s', archive_path)
  # with open(archive_path, 'w'):
  for file_ in manifest.file:
    path = package_dir / file_.relative_path
    print("PATH", path.is_file())


def _CreateArchiveFromPackage() -> None:
  """TODO."""
  package_dir = pathlib.Path(FLAGS.package)
  manifest = pbutil.FromFile(
      package_dir / 'MANIFEST.pbtxt', dpack_pb2.DataPackage())
  VerifyPackageManifest(package_dir, manifest)
  archive_path = (package_dir / f'../{package_dir.name}.dpack.tar.bz2').resolve()
  CreatePackageArchive(package_dir, manifest, archive_path)


def _WriteManifestFile(update: bool) -> None:
  """TODO."""
  updated = False
  package_dir = pathlib.Path(FLAGS.package)
  manifest = CreatePackageManifest(package_dir, FLAGS.exclude)
  manifest_path = package_dir / 'MANIFEST.pbtxt'
  if update and pbutil.ProtoIsReadable(manifest_path, dpack_pb2.DataPackage()):
    old = pbutil.FromFile(manifest_path, dpack_pb2.DataPackage())
    _MergeComments(manifest, old)
    updated = True
  elif manifest_path.is_file():
    raise OSError('Refusing to overwrite MANIFEST.pbtxt file.')
  pbutil.ToFile(manifest, manifest_path)
  if updated:
    logging.info('Updated MANIFEST.pbtxt')
  else:
    logging.info('Created MANIFEST.pbtxt')


def _VerifyPackage() -> None:
  """TODO."""
  package_dir = pathlib.Path(FLAGS.package)
  if not (package_dir / 'MANIFEST.pbtxt').is_file():
    logging.info('No MANIFEST.pbtxt, nothing to do.')
    sys.exit(1)
  manifest = pbutil.FromFile(
      package_dir / 'MANIFEST.pbtxt', dpack_pb2.DataPackage())
  if VerifyPackageManifest(package_dir, manifest):
    logging.info('Package verified. No changes to files in the manifest.')
  else:
    logging.error('Package contains errors.')
    sys.exit(1)


def main(argv) -> None:
  """Main entry point."""
  # Validate flags.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.package:
    raise app.UsageError('--package argument is required.')

  if FLAGS.update and not FLAGS.init:
    logging.warning('--update flag ignored.')

  # Perform the requested action.
  if FLAGS.pack:
    _CreateArchiveFromPackage()
  elif FLAGS.init:
    _WriteManifestFile(update=FLAGS.update)
  else:
    _VerifyPackage()


if __name__ == '__main__':
  app.run(main)
