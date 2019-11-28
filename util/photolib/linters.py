"""This file contains the linter implementations for photolint."""
import csv
import inspect
import os
import pathlib
import sys
import time
import typing
from collections import defaultdict

from labm8.py import app
from labm8.py import shell
from util.photolib import common
from util.photolib import contentfiles
from util.photolib import lintercache
from util.photolib import workspace
from util.photolib import xmp_cache
from util.photolib.proto import photolint_pb2

FLAGS = app.FLAGS
app.DEFINE_boolean("counts", False, "Show only the counts of errors.")
app.DEFINE_boolean("fix_it", False, "Show how to fix it.")

# A global list of all error categories. Every time you add a new linter rule, add it
# here!
ERROR_CATEGORIES = set(
  [
    "dir/empty",
    "dir/not_empty",
    "dir/hierarchy",
    "file/name",
    "file/missing",
    "extension/lowercase",
    "extension/bad",
    "extension/unknown",
    "keywords/third_party",
    "keywords/film_format",
    "keywords/panorama",
    "keywords/inconsistent",
    "keywords/missing",
    "keywords/events",
  ]
)

app.DEFINE_list("ignore", [], "A list of categories to ignore.")

# A map of error categories to error counts.
ERROR_COUNTS: typing.Dict[str, int] = defaultdict(int)


def PrintErrorCounts(end: str = "") -> None:
  """Print the current error counters to stderr."""
  counts = [f"{k}={v}" for k, v in sorted(ERROR_COUNTS.items())]
  counts_str = ", ".join(counts)
  error_str = f"Error counts: {counts_str}." if counts_str else "No errors!"
  print(f"\r\033[Kdone. {error_str}", end=end, file=sys.stderr)


class Error(object):
  """A linter error."""

  def __init__(
    self, relpath: str, category: str, message: str, fix_it: str = None
  ):
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

    if not FLAGS.counts and self.category not in FLAGS.ignore:
      print(
        f"\r\033[K{relpath}:  "
        f"{shell.ShellEscapeCodes.YELLOW}{message}"
        f"{shell.ShellEscapeCodes.END}  [{category}]",
        file=sys.stderr,
      )
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


def GetLinters(
  base_linter: Linter, workspace_root_path: str
) -> typing.List[Linter]:
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
    self, contentfile: contentfiles.Contentfile
  ) -> typing.List[Error]:
    """Run the specified linter on the file.

    Args:
        contentfile: The file to lint.

    Returns:
      A list of zero or more errors.
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

  def __call__(self, contentfile: contentfiles.Contentfile):
    def _CompareMatchToParent(match, parent_directory_name: str):
      (year, month, day), err = contentfiles.get_yyyy_mm_dd(
        parent_directory_name
      )
      if err:  # error will be caught by a directory linter
        return []

      # Compare YYYY-MM-DD against directory name.
      file_year = match.group("year")
      if len(file_year) == 2:
        file_year = f"20{file_year}"

      if (
        year != int(file_year)
        or month != int(match.group("month"))
        or day != int(match.group("day"))
      ):
        return [
          Error(
            contentfile.relpath,
            "dir/hierarchy",
            "file date "
            f"{file_year}-{match.group('month')}-{match.group('day')} "
            "does not match directory date "
            f"{year}-{month:02d}-{day:02d}",
          )
        ]
      return []

    parent_directory_name = contentfile.path.parent.name

    match = common.PHOTO_LIB_PATH_COMPONENTS_RE.match(
      contentfile.filename_noext
    )
    if match:
      return _CompareMatchToParent(match, parent_directory_name)

    match = common.PHOTO_LIB_SCAN_PATH_COMPONENTS_RE.match(
      contentfile.filename_noext
    )
    if match:
      return _CompareMatchToParent(match, parent_directory_name)

    return [Error(contentfile.relpath, "file/name", "invalid file name")]


class ThirdPartyFilename(ThirdPartyFileLinter):
  """Checks that file name matches the expected format."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    errors = []
    topdir = contentfile.path.parent.name

    components = contentfile.filename_noext.split("-")
    if len(components) == 2:
      if components[0] != topdir:
        errors.append(
          Error(
            contentfile.relpath,
            "file/name",
            f"'{components[0]}' should be named '{topdir}'",
          )
        )
      try:
        seq = int(components[1])
      except ValueError:
        return [
          Error(
            contentfile.relpath,
            "file/name",
            f"'{contentfile}' is not a numeric sequence",
          ),
        ]
      try:
        if seq > 1:
          # The length of sequence numbers, including zero-padding.
          seq_len = len(components[1])
          prev = seq - 1

          # Construct the previous name in the sequence.
          filename = (
            f"{components[0]}-{prev:0{seq_len}d}{contentfile.extension}"
          )
          prev_path = contentfile.path.parent / filename
          if not os.path.isfile(prev_path):
            errors.append(
              Error(
                contentfile.relpath, "file/name", f"filename is out-of-sequence"
              )
            )
      except ValueError:
        errors.append(
          Error(contentfile.relpath, "file/name", f"'{seq}' not a number")
        )
    else:
      errors.append(
        Error(
          contentfile.relpath, "file/name", f"should be named '{topdir}-<num>'"
        )
      )

    return errors


class FileExtension(PhotolibFileLinter, ThirdPartyFileLinter):
  """Checks file extensions."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    if contentfile.extension not in common.KNOWN_FILE_EXTENSIONS:
      return []

    errors = []
    lext = contentfile.extension.lower()

    if lext != contentfile.extension:
      labspath = contentfile.abspath[: -len(lext)] + lext
      errors.append(
        Error(
          contentfile.relpath,
          "extension/lowercase",
          "file extension should be lowercase",
          fix_it=(
            f"mv -v '{contentfile.abspath}' '{contentfile.abspath}.tmp' ; "
            f"mv -v '{contentfile.abspath}.tmp' '{labspath}'"
          ),
        )
      )

    if lext not in common.KNOWN_FILE_EXTENSIONS:
      if lext in common.FILE_EXTENSION_SUGGESTIONS:
        suggestion = common.FILE_EXTENSION_SUGGESTIONS[lext]
        errors.append(
          Error(
            contentfile.relpath,
            "extension/bad",
            f"convert {lext} file to {suggestion}",
          )
        )
      else:
        errors.append(
          Error(
            contentfile.relpath, "extension/unknown", "unknown file extension"
          )
        )

    return errors


class PanoramaKeyword(PhotolibFileLinter, ThirdPartyFileLinter):
  """Checks that panorama keywords are set on -Pano files."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    if not contentfile.filename_noext.endswith("-Pano"):
      return []

    errors = []

    if "ATTR|PanoPart" in contentfile.keywords:
      errors.append(
        Error(
          contentfile.relpath,
          "keywords/panorama",
          "keyword 'ATTR >> PanoPart' is set on suspected panorama",
        )
      )

    if "ATTR|Panorama" not in contentfile.keywords:
      errors.append(
        Error(
          contentfile.relpath,
          "keywords/panorama",
          "keyword 'ATTR >> Panorama' not set on suspected panorama",
        )
      )

    return errors


class FilmFormat(PhotolibFileLinter):
  """Checks that 'Film Format' keyword is set on film scans."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    if not common.PHOTO_LIB_SCAN_PATH_COMPONENTS_RE.match(
      contentfile.filename_noext
    ):
      return []

    if not any(k.startswith("ATTR|Film Format") for k in contentfile.keywords):
      return [
        Error(
          contentfile.relpath,
          "keywords/film_format",
          "keyword 'ATTR >> Film Format' not set on film scan",
        )
      ]
    return []


class ThirdPartyInPhotolib(PhotolibFileLinter):
  """Checks that 'third_party' keyword is not set on files in //photos."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    if "ATTR|third_party" in contentfile.keywords:
      return [
        Error(
          contentfile.relpath,
          "keywords/third_party",
          "third_party file should be in //third_party",
        )
      ]
    return []


class ThirdPartyKeywordIsSet(ThirdPartyFileLinter):
  """Checks that 'third_party' keyword is set on files in //third_party."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    if "third_party" not in contentfile.keywords:
      return [
        Error(
          contentfile.relpath,
          "keywords/third_party",
          "files in //third_party should have third_party keyword set",
        )
      ]
    return []


class SingularEvents(PhotolibFileLinter):
  """Checks that only a single 'EVENT' keyword is set."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    num_events = sum(
      1 if k.startswith("EVENT|") else 0 for k in contentfile.keywords
    )

    if num_events > 1:
      events = ", ".join(
        [f"'{k}'" for k in contentfile.keywords if k.startswith("EVENT|")]
      )
      return [
        Error(
          contentfile.relpath,
          "keywords/events",
          f"mutually exclusive keywords = {events}",
        )
      ]
    return []


class DescriptiveKeywords(PhotolibFileLinter):
  """Checks that a file contains descriptive keywords."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    if contentfile.extension not in common.KNOWN_IMG_FILE_EXTENSIONS:
      return []

    errors = []

    # Build a list of "descriptive" keywords, where a descriptive word is not:
    #   * An EVENT keyword.
    #   * The ONDVD attribute.
    descriptive_keywords = [
      k
      for k in contentfile.keywords
      if not k.startswith("EVENT|")
      and not k.startswith("ATTR|")
      and not k.startswith("third_party")
    ]

    # Start with the more specific error before wining about no keywords.
    has_location = any(k.startswith("LOC|") for k in descriptive_keywords)
    is_screenshot = "ATTR|Screenshot" in contentfile.keywords
    is_document = "SUBJ|Document" in contentfile.keywords
    if not has_location and not is_screenshot and not is_document:
      errors.append(
        Error(
          contentfile.relpath,
          "keywords/missing",
          "has no LOC location keywords",
        )
      )
    elif not descriptive_keywords and not is_screenshot and not is_document:
      errors.append(
        Error(
          contentfile.relpath, "keywords/missing", "has no non-event keywords"
        )
      )

    return errors


class MatchingKeywordsOnModifiedFiles(PhotolibFileLinter):
  """Check that keywords are equal on -{HDR,Pano,Edit} files (except pano)."""

  def __call__(self, contentfile: contentfiles.Contentfile):
    if not contentfile.is_composite_file:
      return []

    base_contentfile = contentfile.composite_file_base
    edit_type = "/".join(contentfile.composite_file_types)
    if not base_contentfile:
      return [
        Error(
          contentfile.relpath,
          "file/missing",
          f"could not find {edit_type} base",
        )
      ]

    base_keywords = {
      k
      for k in base_contentfile.keywords
      if k != "ATTR|PanoPart"
      and k != "ATTR|Panorama"
      and k != "ATTR|ONDVD"
      and not k.startswith("PUB|")
    }
    keywords = {
      k
      for k in contentfile.keywords
      if k != "ATTR|Panorama"
      and k != "ATTR|Panorama"
      and k != "ATTR|ONDVD"
      and not k.startswith("PUB|")
    }

    if base_keywords == keywords:
      return []

    errors = []
    extra_keywords = keywords - base_keywords
    if extra_keywords:
      errors.append(
        Error(
          contentfile.relpath,
          "keywords/inconsistent",
          (
            f"{edit_type} file has keywords not found in base "
            f"{base_contentfile.filename}: {','.join(extra_keywords)}"
          ),
        )
      )

    missing_keywords = base_keywords - keywords
    if missing_keywords:
      errors.append(
        Error(
          contentfile.relpath,
          "keywords/inconsistent",
          (
            f"{edit_type} file is missing keywords from base "
            f"{base_contentfile.filename}: {','.join(missing_keywords)}"
          ),
        )
      )

    return errors


# Directory linters.


class DirLinter(Linter):
  """Lint a directory."""

  def __call__(
    self,
    abspath: str,  # pylint: disable=arguments-differ
    workspace_relpath: str,
    dirnames: typing.List[str],
    filenames: typing.List[str],
    files_ignored: bool,
  ) -> None:
    """Lint a directory.

    Args:
        abspath: The absolute path of the directory to lint.
        workspace_relpath: The workspace-relative path.
        dirnames: A list of subdirectories, excluding those which are
            ignored.
        filenames: A list of filenames, excluding those which are ignored.
        files_ignored: True if one or more file in the directory was excluded
          from the list of filenames.
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

  def __call__(
    self,
    abspath: str,
    workspace_relpath: str,
    dirnames: typing.List[str],
    filenames: typing.List[str],
    files_ignored: bool,
  ):
    if not files_ignored and not filenames and not dirnames:
      return [
        Error(
          workspace_relpath,
          "dir/empty",
          "directory is empty, remove it",
          fix_it=f"rmdir '{abspath}'",
        )
      ]
    return []


class ThirdPartyDirname(ThirdPartyDirLinter):
  """Checks that directory name matches the expected format."""

  def __call__(
    self,
    abspath: str,
    workspace_relpath: str,
    dirnames: typing.List[str],
    filenames: typing.List[str],
    files_ignored: bool,
  ):
    del files_ignored
    top_dir = os.path.basename(os.path.dirname(abspath))
    if " " in top_dir or "\t" in top_dir:
      return [Error(workspace_relpath, "dir/hierarchy", "whitespace in path")]
    return []


class DirShouldNotHaveFiles(PhotolibDirLinter):
  """Checks whether a non-leaf directory contains files."""

  def __call__(
    self,
    abspath: str,
    workspace_relpath: str,
    dirnames: typing.List[str],
    filenames: typing.List[str],
    files_ignored: bool,
  ):
    del files_ignored

    # Do nothing if we're in a leaf directory.
    if common.PHOTOLIB_LEAF_DIR_RE.match(workspace_relpath):
      return []

    if not filenames:
      return []

    filelist = " ".join([f"'{abspath}{f}'" for f in filenames])
    return [
      Error(
        workspace_relpath,
        "dir/not_empty",
        "directory should be empty but contains files",
        fix_it=f"rm -rv '{filelist}'",
      )
    ]


class DirHierachy(PhotolibDirLinter):
  """Check that directory hierarchy is correct."""

  def __call__(
    self,
    abspath: str,
    workspace_relpath: str,
    dirnames: str,
    filenames: str,
    files_ignored: bool,
  ):
    del files_ignored
    errors = []

    components = os.path.normpath(workspace_relpath[2:]).split(os.sep)[1:]

    if len(components) >= 1:
      year_1, err = contentfiles.get_yyyy(components[0])
      if err:
        return errors + [err]

    if len(components) >= 2:
      (year_2, month_1), err = contentfiles.get_yyyy_mm(components[1])
      if err:
        return errors + [err]
      if year_1 != year_2:
        errors.append(
          Error(
            workspace_relpath,
            "dir/hierarchy",
            f"years {year_1} and {year_2} do not match",
          )
        )

    if len(components) >= 3:
      (year_3, month_2, _), err = contentfiles.get_yyyy_mm_dd(components[2])
      if err:
        return errors + [err]
      if year_2 != year_3:
        errors.append(
          Error(
            workspace_relpath,
            "dir/hierarchy",
            f"years {year_2} and {year_3} do not match",
          )
        )
      if month_1 != month_2:
        errors.append(
          Error(
            workspace_relpath,
            "dir/hierarchy",
            f"years {month_1} and {month_2} do not match",
          )
        )

    return errors


# Meta-linters.


class Timers(object):
  """Profiling timers."""

  total_seconds: float = 0
  linting_seconds: float = 0
  cached_seconds: float = 0


TIMERS = Timers()


class ToplevelLinter(Linter):
  """A linter for top level directories."""

  __cost__ = 1

  def __init__(
    self,
    workspace_: workspace.Workspace,
    toplevel_dir_relpath: str,
    dirlinters: DirLinter,
    filelinters: FileLinter,
    timers: Timers,
  ):
    super(ToplevelLinter, self).__init__(workspace_)
    self.toplevel_dir = self.workspace.workspace_root / toplevel_dir_relpath
    self.dirlinters = GetLinters(dirlinters, self.workspace)
    self.filelinters = GetLinters(filelinters, self.workspace)
    self.errors_cache = lintercache.LinterCache(self.workspace)
    self.xmp_cache = xmp_cache.XmpCache(self.workspace)
    self.timers = timers

    linter_names = list(
      type(lin).__name__ for lin in self.dirlinters + self.filelinters
    )
    app.Log(
      2, "Running //%s linters: %s", self.toplevel_dir, ", ".join(linter_names)
    )

  def _GetIgnoredNames(self, abspath: str) -> typing.Set[str]:
    """Get the set of file names within a directory to ignore."""
    ignore_file_names = set()

    ignore_file = os.path.join(abspath, common.IGNORE_FILE_NAME)
    if os.path.isfile(ignore_file):
      app.Log(2, "Reading ignore file %s", ignore_file)
      with open(ignore_file) as f:
        for line in f:
          line = line.split("#")[0].strip()
          if line:
            ignore_file_names.add(line)

    return ignore_file_names

  def _LintThisDirectory(
    self,
    abspath: str,
    relpath: str,
    dirnames: typing.List[str],
    all_filenames: typing.List[str],
  ) -> typing.List[Error]:
    """Run linters in this directory."""
    errors = []

    # Strip files and directories which are not to be linted.
    ignored_names = self._GetIgnoredNames(abspath)
    ignored_dirs = common.IGNORED_DIRS.union(ignored_names)
    dirnames = [d for d in dirnames if d not in ignored_dirs]
    ignored_files = common.IGNORED_FILES.union(ignored_names)
    filenames = [f for f in all_filenames if f not in ignored_files]
    files_ignored = len(filenames) != len(all_filenames)

    for linter in self.dirlinters:
      errors += linter(abspath, relpath, dirnames, filenames, files_ignored)

    for filename in filenames:
      contentfile = contentfiles.Contentfile(
        f"{abspath}/{filename}",
        f"{relpath}/{filename}",
        filename,
        self.xmp_cache,
      )
      for linter in self.filelinters:
        errors += linter(contentfile)

    return errors

  def __call__(self, directory: pathlib.Path):
    """Run the linters."""
    start_ = time.time()

    if directory == self.workspace.workspace_root:
      directory = self.toplevel_dir

    directory_str = str(directory.absolute())
    toplevel_str = str(self.toplevel_dir.absolute())

    # Start at the top level.
    if not directory_str.startswith(toplevel_str):
      return

    for abspath, dirnames, filenames in os.walk(directory):
      _start = time.time()
      relpath = self.workspace.GetRelpath(abspath)
      print("\033[KScanning", relpath, end=" ...\r")
      sys.stdout.flush()

      cache_entry = self.errors_cache.GetLinterErrors(abspath, relpath)

      if cache_entry.exists:
        for error in cache_entry.errors:
          ERROR_COUNTS[error.category] += 1
          if not FLAGS.counts:
            print("\r\033[K", error, sep="", file=sys.stderr)
        sys.stderr.flush()

        if FLAGS.counts:
          PrintErrorCounts()

        self.timers.cached_seconds += time.time() - _start
      else:
        errors = self._LintThisDirectory(abspath, relpath, dirnames, filenames)
        self.errors_cache.AddLinterErrors(cache_entry, errors)
        self.timers.linting_seconds += time.time() - _start

    self.timers.total_seconds += time.time() - start_


class WorkspaceLinter(Linter):
  """The master linter for the photolib workspace."""

  __cost__ = 1

  def __init__(self, workspace_: workspace.Workspace):
    self.workspace = workspace_
    self.timers = Timers()

  def __call__(self, directory: pathlib.Path):
    photolib_linter = ToplevelLinter(
      self.workspace, "photos", PhotolibDirLinter, PhotolibFileLinter, TIMERS
    )
    third_party = ToplevelLinter(
      self.workspace,
      "third_party",
      ThirdPartyDirLinter,
      ThirdPartyFileLinter,
      TIMERS,
    )

    photolib_linter(directory)
    third_party(directory)


def Lint(
  workspace_: workspace.Workspace, directory: typing.Optional[pathlib.Path]
):
  """Lint the specified directory and all subdirectories."""
  error_cache = lintercache.LinterCache(workspace_)
  if FLAGS.rm_errors_cache:
    error_cache.Empty()
  xmp_cache.XmpCache(workspace_)
  WorkspaceLinter(workspace_)(directory)


# CSV linters.


class CsvFileLinter(FileLinter):
  pass


class CsvFileDumper(CsvFileLinter):
  """Dump XMP data as CSV."""

  def __init__(self, workspace_: workspace.Workspace, file=sys.stderr):
    self.xmp_cache = xmp_cache.XmpCache(workspace_)
    self.csv_writer = csv.DictWriter(
      file,
      [
        "relpath",
        "camera",
        "lens",
        "iso",
        "shutter_speed",
        "aperture",
        "focal_length_35mm",
        "flash_fired",
        "keywords",
      ],
    )

  def __call__(
    self, contentfile: contentfiles.Contentfile
  ) -> typing.List[Error]:
    """Write the XMP metadata for a file as CSV.

    Returns:
      No errors.
    """
    entry = self.xmp_cache.GetOrCreateXmpCacheEntry(
      contentfile.abspath, contentfile.relpath
    )
    self.csv_writer.writerow(entry.ToDict())
    return []


class CsvDirLinter(DirLinter):
  pass
