"""Module for handling archive files."""
import pathlib
import shutil
import tempfile
import typing
import zipfile

from absl import flags


FLAGS = flags.FLAGS


class UnsupportedArchiveFormat(ValueError):
  """Raised in case an archive has an unsupported file format."""
  pass


class Archive(object):
  """An archive file.

  Provides uniform access unpacked archives when used as a context manager by
  extracting the archive contents to a temporary directory.

  Example:
    >>> with Archive("/tmp/data.zip") as uncompressed_root:
    ...   print(uncompressed_root.iterdir())
    ['a', 'README.txt']

  If the archive is a bazel data dependency, you can use the subclass
  labm8.bazelutil.DataArchive to resolve the absolute path.
  """

  def __init__(self, path: typing.Union[str, pathlib.Path],
               assume_filename: typing.Optional[
                 typing.Union[str, pathlib.Path]] = None):
    """Create an archive.

    Will determine the type of the archive from the suffix, e.g. if path is
    'foo.zip', will treat the file as a zip file. The assume_filename path
    can be used to change the determined type.

    Args:
      path: The path to the data, including the name of the workspace.
      assume_filename: For the purpose of determining the encoding of the
        archive from the file extension, use this name rather than the true
        path.

    Raises:
      FileNotFoundError: If path is not a file.
    """
    self._compressed_path = pathlib.Path(path)
    if not self._compressed_path.is_file():
      raise FileNotFoundError(f"No such file: '{path}'")

    # The path used to determine the type of the archive.
    path_to_determine_type = pathlib.Path(assume_filename or path)
    suffixes = path_to_determine_type.suffixes

    if not suffixes:
      raise UnsupportedArchiveFormat(
          f"Archive '{path_to_determine_type.name}' has no extension")

    if suffixes[-1] == '.zip':
      self._open_function = zipfile.ZipFile
      # TODO(cec): Add support for .tar.bz2, .tar, and .tar.gz.
    else:
      raise UnsupportedArchiveFormat(
          f"Unsupported file extension '{suffixes[-1]}' for archive "
          f"'{path_to_determine_type.name}'")

    # Set in __enter__().
    self._uncompressed_path: typing.Optional[pathlib.Path] = None

  @property
  def path(self) -> pathlib.Path:
    """Return the path of the archive."""
    return self._compressed_path

  def __enter__(self) -> pathlib.Path:
    """Unpack the archive and return the uncompressed path.

    Returns:
      The path of the directory containing the uncompressed archive.
    """
    self._uncompressed_path = pathlib.Path(tempfile.mkdtemp(prefix='phd_'))
    with zipfile.ZipFile(str(self._compressed_path)) as f:
      f.extractall(path=str(self._uncompressed_path))
    return self._uncompressed_path

  def __exit__(self, *args):
    """Exit the scope of the archive.

    This deletes the temporary directory that the archive has been unpacked to.
    """
    shutil.rmtree(self._uncompressed_path)
    self._uncompressed_path = None
