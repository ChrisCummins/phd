"""This file contains the linter implementations for photolint."""
import inspect
import os
import re
import sys
import typing
from collections import defaultdict

from labm8 import app
from labm8 import shell
from util.photolib import common
from util.photolib import lintercache
from util.photolib import workspace
from util.photolib import xmp_cache
from util.photolib.proto import photolint_pb2

FLAGS = app.FLAGS
app.DEFINE_boolean("counts", False, "Show only the counts of errors.")
app.DEFINE_boolean("fix_it", False, "Show how to fix it.")

# A global list of all error categories. Every time you add a new linter rule, add it
# here!
ERROR_CATEGORIES = set([
    "dir/empty",
    "dir/not_empty",
    "dir/hierarchy",
    "file/name",
    "extension/lowercase",
    "extension/bad",
    "extension/unknown",
    "keywords/third_party",
    "keywords/film_format",
    "keywords/panorama",
    "keywords/events",
])

# A map of error categories to error counts.
ERROR_COUNTS: typing.Dict[str, int] = defaultdict(int)


def PrintErrorCounts() -> None:
  """Print the current error counters to stderr."""
  counts = [f"{k} = {v}" for k, v in sorted(ERROR_COUNTS.items())]
  counts_str = ", ".join(counts)
  print(f"\r{counts_str}", end="", file=sys.stderr)


class Error(object):
  """A linter error."""

  def __init__(self,
               relpath: str,
               category: str,
               message: str,
               fix_it: str = None):
    """Report an error.

    If --counts flag was passed, this updates the running totals of error counts.
    If --fix_it flag was passed, the command is printed to stdout.
    All other output is to stderr.
    """
    assert category in ERROR_CATEGORIES

    self.relpath = relpath
    self.category = category
    self.message = message
    self.fix_it = fix_it

    ERROR_COUNTS[category] += 1

    if not FLAGS.counts:
      print(
          f'{relpath}:  '
          f'{shell.ShellEscapeCodes.YELLOW}{message}'
          f'{shell.ShellEscapeCodes.END}  [{category}]',
          file=sys.stderr)
      sys.stderr.flush()

    if FLAGS.fix_it and fix_it:
      print(fix_it)

  def ToProto(self) -> photolint_pb2.PhotolintFileError:
    """Instantiate a PhotolintFileError message for this error.

    Returns:
      A PhotolintFileError message instance.
    """
    error = photolint_pb2.PhotolintFileError()
    error.workspace_relative_path = self.relpath
    error.category = self.category
    error.message = self.message
    error.shell_command_to_fix_it = self.fix_it


class Linter(object):
  """A linter is an object which checks for errors.

  When called with some input, verify the input and return a list of errors.
  """

  def __init__(self, workspace_: workspace.Workspace):
    self.workspace = workspace_
    self.errors_cache = lintercache.LinterCache(self.workspace)
    self.xmp_cache = xmp_cache.XmpCache(self.workspace)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError("abstract class")


def GetLinters(base_linter: Linter,
               workspace_root_path: str) -> typing.List[Linter]:
  """Return a list of linters to run."""

  def _IsRunnableLinter(obj):
    """Return true if obj is a runnable linter."""
    if not inspect.isclass(obj):
      return False
    if not issubclass(obj, base_linter):
      return False
    if obj == base_linter:  # Don't include the base_linter.
      return False
    return True

  members = inspect.getmembers(sys.modules[__name__], _IsRunnableLinter)
  return [member[1](workspace_root_path) for member in members]


# File linters.


class FileLinter(Linter):
  """Lint a file."""

  def __call__(
      self,
      abspath: str,  # pylint: disable=arguments-differ
      workspace_relpath: str,
      filename: str) -> None:
    """

    Args:
        abspath: The absolute path of the file to lint.
        workspace_relpath: The workspace-relative path of the file to lint.
        filename: The basename of the file to lint.
    """
    raise NotImplementedError("abstract class")


class PhotolibFileLinter(FileLinter):  # pylint: disable=abstract-method
  """Lint a file in //photos."""
  pass


class ThirdPartyFileLinter(FileLinter):  # pylint: disable=abstract-method
  """Lint a file in //third_party."""
  pass


class PhotolibFilename(PhotolibFileLinter):
  """Checks that file name matches one of expected formats."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    filename_noext = os.path.splitext(filename)[0]
    if re.match(common.PHOTO_LIB_PATH_COMPONENTS_RE, filename_noext):
      # TODO(cec): Compare YYYY-MM-DD against directory name.
      return []

    if re.match(common.PHOTO_LIB_SCAN_PATH_COMPONENTS_RE, filename_noext):
      # TODO(cec): Compare YYYY-MM-DD against directory name.
      return []

    return [Error(workspace_relpath, "file/name", "invalid file name")]


class ThirdPartyFilename(ThirdPartyFileLinter):
  """Checks that file name matches the expected format."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    errors = []
    topdir = os.path.basename(os.path.dirname(abspath))
    filename_noext, ext = os.path.splitext(filename)

    components = filename_noext.split("-")
    if len(components) == 2:
      if components[0] != topdir:
        errors.append(
            Error(workspace_relpath, "file/name",
                  f"'{components[0]}' should be '{topdir}'"))
      try:
        seq = int(components[1])
        if seq > 1:
          # The length of sequence numbers, including zero-padding.
          seq_len = len(components[1])
          prev = seq - 1

          # Construct the previous name in the sequence.
          filename = f"{components[0]}-{prev:0{seq_len}d}{ext}"
          prev_path = os.path.join(os.path.dirname(abspath), filename)
          if not os.path.isfile(prev_path):
            errors.append(
                Error(workspace_relpath, "file/name",
                      f"filename is out-of-sequence"))
      except ValueError:
        errors.append(
            Error(workspace_relpath, "file/name", f"'{seq}' not a number"))
    else:
      errors.append(
          Error(workspace_relpath, "file/name", f"should be '{topdir}-<num>'"))

    return errors


class FileExtension(PhotolibFileLinter, ThirdPartyFileLinter):
  """Checks file extensions."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    errors = []
    ext = os.path.splitext(filename)[-1]
    lext = ext.lower()

    if lext != ext:
      labspath = abspath[:-len(ext)] + lext
      errors.append(
          Error(workspace_relpath,
                "extension/lowercase",
                "file extension should be lowercase",
                fix_it=(f"mv -v '{abspath}' '{abspath}.tmp' ; "
                        f"mv -v '{abspath}.tmp' '{labspath}'")))

    if lext not in common.KNOWN_FILE_EXTENSIONS:
      if lext == ".jpeg":
        jabspath = abspath[:-len(ext)] + ".jpg"
        errors.append(
            Error(workspace_relpath,
                  "extension/bad",
                  f"convert {lext} file to .jpg",
                  fix_it=f"mv -v '{abspath}' '{jabspath}'"))
      if lext in common.FILE_EXTENSION_SUGGESTIONS:
        suggestion = common.FILE_EXTENSION_SUGGESTIONS[lext]
        errors.append(
            Error(workspace_relpath, "extension/bad",
                  f"convert {lext} file to {suggestion}"))
      else:
        errors.append(
            Error(workspace_relpath, "extension/unknown",
                  "unknown file extension"))

    return errors


class PanoramaKeyword(PhotolibFileLinter, ThirdPartyFileLinter):
  """Checks that panorama keywords are set on -Pano files."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    if "-Pano" not in filename:
      return []

    errors = []

    keywords = self.xmp_cache.GetLightroomKeywords(abspath, workspace_relpath)
    if "ATTR|PanoPart" not in keywords:
      errors.append(
          Error(workspace_relpath, "keywords/panorama",
                "keyword 'ATTR >> PanoPart' not set on suspected panorama"))

    if "ATTR|Panorama" not in keywords:
      errors.append(
          Error(workspace_relpath, "keywords/panorama",
                "keyword 'ATTR >> Panorama' not set on suspected panorama"))

    return errors


class FilmFormat(PhotolibFileLinter):
  """Checks that 'Film Format' keyword is set on film scans."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    filename_noext = os.path.splitext(filename)[0]
    if not common.PHOTO_LIB_SCAN_PATH_COMPONENTS_RE.match(filename_noext):
      return []

    keywords = self.xmp_cache.GetLightroomKeywords(abspath, workspace_relpath)
    if not any(k.startswith("ATTR|Film Format") for k in keywords):
      return [
          Error(workspace_relpath, "keywords/film_format",
                "keyword 'ATTR >> Film Format' not set on film scan")
      ]
    return []


class ThirdPartyInPhotolib(PhotolibFileLinter):
  """Checks that 'third_party' keyword is not set on files in //photos."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    keywords = self.xmp_cache.GetLightroomKeywords(abspath, workspace_relpath)
    if "ATTR|third_party" in keywords:
      return [
          Error(workspace_relpath, "keywords/third_party",
                "third_party file should be in //third_party")
      ]
    return []


class ThirdPartyKeywordIsSet(ThirdPartyFileLinter):
  """Checks that 'third_party' keyword is set on files in //third_party."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    keywords = self.xmp_cache.GetLightroomKeywords(abspath, workspace_relpath)
    if "third_party" not in keywords:
      return [
          Error(workspace_relpath, "keywords/third_party",
                "files in //third_party should have third_party keyword set")
      ]
    return []


class SingularEvents(PhotolibFileLinter):
  """Checks that only a single 'EVENT' keyword is set."""

  def __call__(self, abspath: str, workspace_relpath: str, filename: str):
    keywords = self.xmp_cache.GetLightroomKeywords(abspath, workspace_relpath)
    num_events = sum(1 if k.startswith("EVENT|") else 0 for k in keywords)

    if num_events > 1:
      events = ", ".join([f"'{k}'" for k in keywords if k.startswith("EVENT|")])
      return [
          Error(workspace_relpath, "keywords/events",
                f"mutually exclusive keywords = {events}")
      ]
    return []


# Directory linters.


class DirLinter(Linter):
  """Lint a directory."""

  def __call__(
      self,
      abspath: str,  # pylint: disable=arguments-differ
      workspace_relpath: str,
      dirnames: typing.List[str],
      filenames: typing.List[str]) -> None:
    """Lint a directory.

    Args:
        abspath: The absolute path of the directory to lint.
        workspace_relpath: The workspace-relative path.
        dirnames: A list of subdirectories, excluding those which are
            ignored.
        filenames: A list of filenames, excluding those which are ignored.
    """
    pass


class PhotolibDirLinter(DirLinter):  # pylint: disable=abstract-method
  """Lint a directory in //photos."""
  pass


class ThirdPartyDirLinter(DirLinter):  # pylint: disable=abstract-method
  """Lint a directory in //third_party."""
  pass


class DirEmpty(PhotolibDirLinter, ThirdPartyDirLinter):
  """Checks whether a directory is empty."""

  def __init__(self, *args, **kwargs):
    PhotolibDirLinter.__init__(self, *args, **kwargs)

  def __call__(self, abspath: str, workspace_relpath: str,
               dirnames: typing.List[str], filenames: typing.List[str]):
    if not filenames and not dirnames:
      return [
          Error(workspace_relpath,
                "dir/empty",
                "directory is empty, remove it",
                fix_it=f"rmdir '{abspath}'")
      ]
    return []


class ThirdPartyDirname(ThirdPartyDirLinter):
  """Checks that directory name matches the expected format."""

  def __call__(self, abspath: str, workspace_relpath: str,
               dirnames: typing.List[str], filenames: typing.List[str]):
    top_dir = os.path.basename(os.path.dirname(abspath))
    if " " in top_dir or "\t" in top_dir:
      return [Error(workspace_relpath, "dir/hierarchy", "whitespace in path")]
    return []


class DirShouldNotHaveFiles(PhotolibDirLinter):
  """Checks whether a non-leaf directory contains files."""

  def __call__(self, abspath: str, workspace_relpath: str,
               dirnames: typing.List[str], filenames: typing.List[str]):
    # Do nothing if we're in a leaf directory.
    if common.PHOTOLIB_LEAF_DIR_RE.match(workspace_relpath):
      return []

    if not filenames:
      return []

    filelist = " ".join([f"'{abspath}{f}'" for f in filenames])
    return [
        Error(workspace_relpath,
              "dir/not_empty",
              "directory should be empty but contains files",
              fix_it=f"rm -rv '{filelist}'")
    ]


class DirHierachy(PhotolibDirLinter):
  """Check that directory hierarchy is correct."""

  def __call__(self, abspath: str, workspace_relpath: str, dirnames: str,
               filenames: str):
    errors = []

    def get_yyyy(string: str) -> typing.Optional[int]:
      """Parse string or show error. Returns None in case of error."""
      try:
        if 1900 <= int(string) <= 2100:
          return int(string)
        errors.append(
            Error(workspace_relpath, "dir/hiearchy",
                  f"year '{string}' out of range"))
      except ValueError:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  "'{string}' should be a four digit year"))
      return None

    def get_mm(string: str) -> typing.Optional[int]:
      """Parse string or show error. Returns None in case of error."""
      try:
        if 1 <= int(string) <= 12:
          return int(string)
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"month '{string}' out of range"))
      except ValueError:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be a two digit month"))
      return None

    def get_dd(string: str) -> typing.Optional[int]:
      """Parse string or show error. Returns None in case of error."""
      try:
        if 1 <= int(string) <= 31:
          return int(string)
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"day '{string}' out of range"))
      except ValueError:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be a two digit day"))
      return None

    def get_yyyy_mm(
        string: str
    ) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
      """Parse string or show error. Returns None in case of error."""
      string_components = string.split("-")
      if len(string_components) == 2:
        year = get_yyyy(string_components[0])
        month = get_mm(string_components[1])
        if year and month:
          return year, month
      else:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be YYYY-MM"))
      return None, None

    def get_yyyy_mm_dd(
        string: str) -> typing.Tuple[typing.Optional[int], typing.
                                     Optional[int], typing.Optional[int]]:
      """Parse string or show error. Returns None in case of error."""
      string_components = string.split("-")
      if len(string_components) == 3:
        year = get_yyyy(string_components[0])
        month = get_mm(string_components[1])
        day = get_dd(string_components[2])
        if year and month and day:
          return year, month, day
      else:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be YYYY-MM-DD"))
      return None, None

    components = os.path.normpath(workspace_relpath[2:]).split(os.sep)[1:]

    if len(components) >= 1:
      year_1 = get_yyyy(components[0])
      if not year_1:
        return errors

    if len(components) >= 2:
      year_2, month_1 = get_yyyy_mm(components[1])
      if not year_2:
        return errors
      if year_1 != year_2:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"years {year_1} and {year_2} do not match"))

    if len(components) >= 3:
      year_3, month_2, _ = get_yyyy_mm_dd(components[2])
      if not year_3:
        return errors
      if year_2 != year_3:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"years {year_2} and {year_3} do not match"))
      if month_1 != month_2:
        errors.append(
            Error(workspace_relpath, "dir/hierarchy",
                  f"years {month_1} and {month_2} do not match"))

    return errors


class ModifiedKeywords(PhotolibDirLinter):
  """Check that keywords are equal on -{HDR,Pano,Edit} files (except pano)."""

  def __call__(self, abspath: str, workspace_relpath: str, dirnames: str,
               filenames: str):
    # TODO(cec): Check that keywords are equal on -{HDR,Pano,Editr} files.
    return []
