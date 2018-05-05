"""Dpack creates structured packages of data files.

Dpack creates an archive of a directory with a manifest. The manifest describes
each of the files within the directory. Use dpack to create consistent and well
documented collections of data files.
"""
import fnmatch
import hashlib

import pathlib
import typing
from absl import app
from absl import flags
from absl import logging

from lib.dpack.proto import dpack_pb2
from lib.labm8 import crypto
from lib.labm8 import fs

FLAGS = flags.FLAGS

flags.DEFINE_string('package_dir', None,
                    'The path of the directory to package.')
flags.DEFINE_list('excludes', [],
                  'A list of patterns to exclude from the package. Supports '
                  'UNIX-style globbing: *,?,[],[!].')

flags.register_validator(
    'package_dir',
    lambda path: pathlib.Path(path).is_dir(),
    message='--package_dir must be a directory.')

# A list of filename patterns to exclude from all data packages.
ALWAYS_EXCLUDE_PATTERNS = [
  '.DS_Store',
]


def _Md5sumFile(path):
  m = hashlib.md5()
  with open(path, 'rb') as infile:
    m.update(infile.read())
  return m.hexdigest()


def WritePackageManifest(package: dpack_pb2.DataPackage, path: pathlib.Path):
  """

  Args:
    package:
    path:

  Returns:

  """
  with open(path, 'w') as f:
    for entry in package.files:
      print(entry.relpath, entry.size, entry.checksum, entry.description, sep='\t')


def _GetPackageContents(package_root: pathlib.Path,
                        exclude_patterns: typing.List[str]) -> typing.List[str]:
  exclude_patterns = set(exclude_patterns + ALWAYS_EXCLUDE_PATTERNS)
  contents = []
  for path in fs.lsfiles(package_root, recursive=True):
    for pattern in exclude_patterns:
      if fnmatch.fnmatch(path, pattern):
        logging.info("excluding path '%s' which matched pattern '%s'",
                     path, pattern)
        break
    else:
      contents.append(pathlib.Path(path))
  return contents


def _SetDataPackageFileAttributes(package_root: pathlib.Path,
                                  relpath: pathlib.Path,
                                  f: dpack_pb2.DataPackageFile) -> None:
  abspath = package_root / relpath
  f.relative_path = str(relpath)
  f.size_in_bytes = abspath.stat().st_size
  f.comment = f.comment or ''
  f.checksum = crypto.sha256_file(abspath)
  f.checksum_algo = dpack_pb2.DataPackageFile.SHA256


def _MergePackageManifest(manifest: dpack_pb2.DataPackage,
                          package_root: pathlib.Path,
                          paths: typing.List[str]):
  manifest.comment = manifest.comment or ''
  files = {f.relative_path: f for f in manifest.file}
  for path in paths:
    # Create a new file entry if it is new, else update.
    f = files[path] if path in files else manifest.file.add()
    _SetDataPackageFileAttributes(package_root, path, f)


def CreatePackageManifest(package_root: pathlib.Path,
                          exclude_patterns: typing.List[str] = []):
  manifest_path = package_root / 'MANIFEST.txt'
  if manifest_path.is_file():
    raise OSError('Refusing to overwrite MANIFEST.txt file.')

  manifest = dpack_pb2.DataPackage()
  _MergePackageManifest(manifest, package_root,
                        _GetPackageContents(package_root, exclude_patterns))
  print(manifest)


def CreateArchiveFromPackage(package_root: pathlib.Path):
  pass


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.package_dir:
    raise app.UsageError('')

  CreatePackageManifest(pathlib.Path(FLAGS.package_dir))


if __name__ == '__main__':
  app.run(main)
