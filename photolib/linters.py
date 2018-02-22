"""This file contains the linter implementations for photolint."""
import os
import re
import sys
import typing
from collections import defaultdict

from absl import flags

from photolib import lightroom
from photolib import util

FLAGS = flags.FLAGS
flags.DEFINE_boolean("counts", False, "Show only the counts of errors.")
flags.DEFINE_boolean("fix_it", False, "Show how to fix it.")

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


def error(relpath: str, category: str, message: str, fix_it: str = None):
    """Report an error.

    If --counts flag was passed, this updates the running totals of error counts.
    If --fix_it flag was passed, the command is printed to stdout.
    All other output is to stderr.
    """
    assert category in ERROR_CATEGORIES

    ERROR_COUNTS[category] += 1

    if FLAGS.counts:
        counts = [f"{k} = {v}" for k, v in sorted(ERROR_COUNTS.items())]
        counts_str = ", ".join(counts)
        print(f"\r{counts_str}", end="", file=sys.stderr)
    else:
        print(f"{relpath}:  {util.Colors.YELLOW}{message}{util.Colors.END}  [{category}]",
              file=sys.stderr)

    if FLAGS.fix_it and fix_it:
        print(fix_it)


class Linter(object):
    """A linter is a reusable class which

    'n' is the number of files in the workspace
    'm' is the size of a file in the workspace

    Costs:
        0-9: O(1)
        10-19: O(n)
        20-29: O(n * m)
        30-39: O(n ** 2)
    """
    __cost__ = -1

    def __init__(self):
        assert self.__cost__ > 0 and self.__cost__ <= 100

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("abstract class")


# File linters.

class FileLinter(Linter):
    """Lint a file."""

    def __call__(self, abspath: str,  # pylint: disable=arguments-differ
                 workspace_relpath: str, filename: str) -> None:
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


class GalleryFileLinter(FileLinter):  # pylint: disable=abstract-method
    """Lint a file in //gallery."""
    pass


class PhotolibFilename(PhotolibFileLinter):
    """Checks that file name matches one of expected formats."""
    __cost__ = 10

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        filename_noext = os.path.splitext(filename)[0]
        if re.match(util.PHOTO_LIB_PATH_COMPONENTS_RE, filename_noext):
            # TODO(cec): Compare YYYY-MM-DD against directory name.
            return

        if re.match(util.PHOTO_LIB_SCAN_PATH_COMPONENTS_RE, filename_noext):
            # TODO(cec): Compare YYYY-MM-DD against directory name.
            return

        error(workspace_relpath, "file/name", "invalid file name")


class GalleryFilename(GalleryFileLinter):
    """Checks that file name matches one of expected formats."""
    __cost__ = 10

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        if " " in filename or "\t" in filename:
            error(workspace_relpath, "file/name", "invalid file name")


class FileExtension(PhotolibFileLinter, GalleryFileLinter):
    """Checks file extensions."""
    __cost__ = 10

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        ext = os.path.splitext(filename)[-1]
        lext = ext.lower()

        if lext != ext:
            labspath = abspath[:-len(ext)] + lext
            error(workspace_relpath, "extension/lowercase",
                  "file extension should be lowercase",
                  fix_it=f"mv -v '{abspath}' '{labspath}'")

        if lext not in util.KNOWN_FILE_EXTENSIONS:
            if lext == ".jpeg":
                jabspath = abspath[:-len(ext)] + ".jpg"
                error(workspace_relpath, "extension/bad",
                      f"convert {lext} file to .jpg",
                      fix_it=f"mv -v '{abspath}' '{jabspath}'")
            if lext in util.FILE_EXTENSION_SUGGESTIONS:
                suggestion = util.FILE_EXTENSION_SUGGESTIONS[lext]
                error(workspace_relpath, "extension/bad",
                      f"convert {lext} file to {suggestion}")
            else:
                error(workspace_relpath, "extension/unknown",
                      "unknown file extension")


class PanoramaKeyword(PhotolibFileLinter, GalleryFileLinter):
    """Checks that panorama keywords are set on -Pano files."""
    __cost__ = 20

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        if "-Pano" not in filename:
            return

        keywords = lightroom.get_lightroom_keywords(abspath)
        if "ATTR|PanoPart" not in keywords:
            error(workspace_relpath, "keywords/panorama",
                  "keyword 'ATTR >> PanoPart' not set on suspected panorama")

        if "ATTR|Panorama" not in keywords:
            error(workspace_relpath, "keywords/panorama",
                  "keyword 'ATTR >> Panorama' not set on suspected panorama")


class FilmFormat(PhotolibFileLinter):
    """Checks that 'Film Format' keyword is set on film scans."""
    __cost__ = 20

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        filename_noext = os.path.splitext(filename)[0]
        if not util.PHOTO_LIB_SCAN_PATH_COMPONENTS_RE.match(filename_noext):
            return

        keywords = lightroom.get_lightroom_keywords(abspath)
        if not any(k.startswith("ATTR|Film Format") for k in keywords):
            error(workspace_relpath, "keywords/film_format",
                  "keyword 'ATTR >> Film Format' not set on film scan")


class ThirdPartyInPhotolib(PhotolibFileLinter):
    """Checks that 'third_party' keyword is not set on files in //photos."""
    __cost__ = 29

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        keywords = lightroom.get_lightroom_keywords(abspath)
        if "ATTR|third_party" in keywords:
            error(workspace_relpath, "keywords/third_party",
                  "third_party file should be in //gallery")


class ThirdPartyInGallery(GalleryFileLinter):
    """Checks that 'third_party' keyword is set on files in //gallery."""
    __cost__ = 29

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        keywords = lightroom.get_lightroom_keywords(abspath)
        if "ATTR|third_party" not in keywords:
            error(workspace_relpath, "keywords/third_party",
                  "files in //gallery should have third_party keyword set")


class SingularEvents(PhotolibFileLinter):
    """Checks that only a single 'EVENT' keyword is set."""
    __cost__ = 20

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        keywords = lightroom.get_lightroom_keywords(abspath)
        num_events = sum(1 if k.startswith("EVENT|") else 0 for k in keywords)

        if num_events > 1:
            events = ", ".join(
                [f"'{k}'" for k in keywords if k.startswith("EVENT|")])
            error(workspace_relpath, "keywords/events",
                  f"mutually exclusive keywords = {events}")


class ModifiedKeywords(PhotolibFileLinter):
    """Check that keywords are equal on -{HDR,Pano,Edit} files (except pano)."""
    __cost__ = 30

    def __call__(self, abspath: str, workspace_relpath: str, filename: str):
        # TODO(cec):
        pass


# Directory linters.

class DirLinter(Linter):
    """Lint a directory."""

    def __call__(self, abspath: str,  # pylint: disable=arguments-differ
                 workspace_relpath: str, dirnames: typing.List[str],
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


class GalleryDirLinter(DirLinter):  # pylint: disable=abstract-method
    """Lint a directory in //gallery."""
    pass


class DirEmpty(PhotolibDirLinter, GalleryDirLinter):
    """Checks whether a directory is empty."""
    __cost__ = 10

    def __call__(self, abspath: str, workspace_relpath: str,
                 dirnames: typing.List[str], filenames: typing.List[str]):
        if not filenames and not dirnames:
            error(workspace_relpath, "dir/empty",
                  "directory is empty, remove it", fix_it=f"rm -rv '{abspath}'")


class DirShouldNotHaveFiles(PhotolibDirLinter):
    """Checks whether a non-leaf directory contains files."""
    __cost__ = 10

    def __call__(self, abspath: str, workspace_relpath: str,
                 dirnames: typing.List[str], filenames: typing.List[str]):
        # Do nothing if we're in a leaf directory.
        if util.PHOTOLIB_LEAF_DIR_RE.match(workspace_relpath):
            return

        if not filenames:
            return

        filelist = " ".join([f"'{abspath}{f}'" for f in filenames])
        error(workspace_relpath, "dir/not_empty",
              "directory should be empty but contains files",
              fix_it=f"rm -rv '{filelist}'")


class DirHierachy(PhotolibDirLinter):
    """Check that directory hierarchy is correct."""
    __cost__ = 10

    @classmethod
    def get_yyyy(
            cls, string: str, workspace_relpath: str) -> typing.Optional[int]:
        """Parse string or show error. Returns None in case of error."""
        try:
            year = int(string)
            if 1900 <= year <= 2100:
                return year
            error(workspace_relpath, "dir/hiearchy",
                  f"year '{year}' out of range")
        except ValueError:
            error(workspace_relpath, "dir/hierarchy",
                  "'{string}' should be a four digit year")
        return None

    @classmethod
    def get_mm(
            cls, string: str, workspace_relpath: str) -> typing.Optional[int]:
        """Parse string or show error. Returns None in case of error."""
        try:
            month = int(string)
            if 1 <= month <= 12:
                return month
            error(workspace_relpath, "dir/hierarchy",
                  f"month '{month}' out of range")
        except ValueError:
            error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be a two digit month")
        return None

    @classmethod
    def get_dd(
            cls, string: str, workspace_relpath: str) -> typing.Optional[int]:
        """Parse string or show error. Returns None in case of error."""
        try:
            day = int(string)
            if 1 <= day <= 31:
                return day
            error(workspace_relpath, "dir/hierarchy",
                  f"day '{day}' out of range")
        except ValueError:
            error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be a two digit day")
        return None

    @classmethod
    def get_yyyy_mm(
            cls, string: str,
            workspace_relpath: str) -> typing.Tuple[typing.Optional[int],
                                                    typing.Optional[int]]:
        """Parse string or show error. Returns None in case of error."""
        components = string.split("-")
        if len(components) == 2:
            year = cls.get_yyyy(components[0], workspace_relpath)
            month = cls.get_mm(components[1], workspace_relpath)
            if year and month:
                return year, month
        else:
            error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be YYYY-MM")
        return None, None

    @classmethod
    def get_yyyy_mm_dd(
            cls, string: str,
            workspace_relpath: str) -> typing.Tuple[typing.Optional[int],
                                                    typing.Optional[int],
                                                    typing.Optional[int]]:
        """Parse string or show error. Returns None in case of error."""
        components = string.split("-")
        if len(components) == 3:
            year = cls.get_yyyy(components[0], workspace_relpath)
            month = cls.get_mm(components[1], workspace_relpath)
            day = cls.get_dd(components[2], workspace_relpath)
            if year and month and day:
                return year, month, day
        else:
            error(workspace_relpath, "dir/hierarchy",
                  f"'{string}' should be YYYY-MM-DD")
        return None, None

    def __call__(self, abspath: str, workspace_relpath: str,
                 dirnames: str, filenames: str):
        components = os.path.normpath(workspace_relpath[2:]).split(os.sep)[1:]

        if len(components) >= 1:
            year_1 = self.get_yyyy(components[0], workspace_relpath)
            if not year_1:
                return

        if len(components) >= 2:
            year_2, month_1 = self.get_yyyy_mm(components[1], workspace_relpath)
            if not year_2:
                return
            if year_1 != year_2:
                error(workspace_relpath, "dir/hierarchy",
                      f"years {year_1} and {year_2} do not match")

        if len(components) >= 3:
            year_3, month_2, _ = self.get_yyyy_mm_dd(
                components[2], workspace_relpath)
            if not year_3:
                return
            if year_2 != year_3:
                error(workspace_relpath, "dir/hierarchy",
                      f"years {year_2} and {year_3} do not match")
            if month_1 != month_2:
                error(workspace_relpath, "dir/hierarchy",
                      f"years {month_1} and {month_2} do not match")
