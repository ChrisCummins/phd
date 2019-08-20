"""A linter for ensuring that a Photo Library is organized correctly."""
import os
import sys
import time
import typing

from labm8 import app
from labm8 import humanize
from util.photolib import common
from util.photolib import contentfiles
from util.photolib import lintercache
from util.photolib import linters
from util.photolib import workspace
from util.photolib import xmp_cache

FLAGS = app.FLAGS
app.DEFINE_string("workspace", "/workspace", "Path to workspace root")
app.DEFINE_boolean("profile", False, "Print profiling timers on completion.")
app.DEFINE_boolean("rm_errors_cache", False,
                   "If true, empty the errors cache piror to running.")


class Timers(object):
  """Profiling timers."""
  total_seconds: float = 0
  linting_seconds: float = 0
  cached_seconds: float = 0


TIMERS = Timers()


class ToplevelLinter(linters.Linter):
  """A linter for top level directories."""
  __cost__ = 1

  def __init__(self, workspace_: workspace.Workspace, toplevel_dir_relpath: str,
               dirlinters: typing.List[linters.DirLinter],
               filelinters: typing.List[linters.FileLinter]):
    super(ToplevelLinter, self).__init__(workspace_)
    self.toplevel_dir = self.workspace.workspace_root / toplevel_dir_relpath
    self.dirlinters = linters.GetLinters(dirlinters, self.workspace)
    self.filelinters = linters.GetLinters(filelinters, self.workspace)
    self.errors_cache = lintercache.LinterCache(self.workspace)
    self.xmp_cache = xmp_cache.XmpCache(self.workspace)

    linter_names = list(
        type(lin).__name__ for lin in self.dirlinters + self.filelinters)
    app.Log(2, "Running //%s linters: %s", self.toplevel_dir,
            ", ".join(linter_names))

  def _GetIgnoredNames(self, abspath: str) -> typing.Set[str]:
    """Get the set of file names within a directory to ignore."""
    ignore_file_names = set()

    ignore_file = os.path.join(abspath, common.IGNORE_FILE_NAME)
    if os.path.isfile(ignore_file):
      app.Log(2, 'Reading ignore file %s', ignore_file)
      with open(ignore_file) as f:
        for line in f:
          line = line.split('#')[0].strip()
          if line:
            ignore_file_names.add(line)

    return ignore_file_names

  def _LintThisDirectory(
      self, abspath: str, relpath: str, dirnames: typing.List[str],
      all_filenames: typing.List[str]) -> typing.List[linters.Error]:
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
      contentfile = contentfiles.Contentfile(f"{abspath}/{filename}",
                                             f"{relpath}/{filename}", filename,
                                             self.xmp_cache)
      for linter in self.filelinters:
        errors += linter(contentfile)

    return errors

  def __call__(self, *args, **kwargs):
    """Run the linters."""
    start_ = time.time()

    working_dir = self.workspace.workspace_root / self.toplevel_dir
    for abspath, dirnames, filenames in os.walk(working_dir):
      _start = time.time()
      relpath = self.workspace.GetRelpath(abspath)

      cache_entry = self.errors_cache.GetLinterErrors(abspath, relpath)

      if cache_entry.exists:
        for error in cache_entry.errors:
          linters.ERROR_COUNTS[error.category] += 1
          if not FLAGS.counts:
            print(error, file=sys.stderr)
        sys.stderr.flush()

        if FLAGS.counts:
          linters.PrintErrorCounts()

        TIMERS.cached_seconds += time.time() - _start
      else:
        errors = self._LintThisDirectory(abspath, relpath, dirnames, filenames)
        self.errors_cache.AddLinterErrors(cache_entry, errors)
        TIMERS.linting_seconds += time.time() - _start

    TIMERS.total_seconds += time.time() - start_


class WorkspaceLinter(linters.Linter):
  """The master linter for the photolib workspace."""
  __cost__ = 1

  def __call__(self, *args, **kwargs):
    photolib_linter = ToplevelLinter(self.workspace, "photos",
                                     linters.PhotolibDirLinter,
                                     linters.PhotolibFileLinter)
    third_party = ToplevelLinter(self.workspace, "third_party",
                                 linters.ThirdPartyDirLinter,
                                 linters.ThirdPartyFileLinter)

    photolib_linter()
    third_party()


def main(argv):  # pylint: disable=missing-docstring
  del argv
  abspath = workspace.find_workspace_rootpath(
      os.path.expanduser(FLAGS.workspace))
  if not abspath:
    print(f"Cannot find workspace in '{FLAGS.workspace}'", file=sys.stderr)
    sys.exit(1)

  workspace_ = workspace.Workspace(abspath)
  error_cache = lintercache.LinterCache(workspace_)
  if FLAGS.rm_errors_cache:
    error_cache.Empty()
  xmp_cache.XmpCache(workspace_)
  WorkspaceLinter(workspace_)()

  # Print the carriage return once we've done updating the counts line.
  if FLAGS.counts:
    if linters.ERROR_COUNTS:
      print("", file=sys.stderr)
  else:
    linters.PrintErrorCounts(end="\n")

  # Print the profiling timers once we're done.
  if FLAGS.profile:
    total_time = TIMERS.total_seconds
    linting_time = TIMERS.linting_seconds
    cached_time = TIMERS.cached_seconds
    overhead = total_time - linting_time - cached_time

    print(
        f'timings: linting={humanize.Duration(linting_time)} '
        f'({linting_time / total_time:.1%}), '
        f'cached={humanize.Duration(cached_time)} '
        f'({cached_time / total_time:.1%}), '
        f'overhead={humanize.Duration(overhead)} '
        f'({overhead / total_time:.1%}), '
        f'total={humanize.Duration(total_time)}.',
        file=sys.stderr)


if __name__ == "__main__":
  try:
    app.RunWithArgs(main)
  except KeyboardInterrupt:
    print("interrupt")
    sys.exit(1)
