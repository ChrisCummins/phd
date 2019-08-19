"""A linter for ensuring that a Photo Library is organized correctly."""
import os
import sys
import time
import typing

from labm8 import app
from util.photolib import common
from util.photolib import lintercache
from util.photolib import linters
from util.photolib import workspace
from util.photolib import xmp_cache

FLAGS = app.FLAGS
app.DEFINE_string("workspace", "/workspace", "Path to workspace root")
app.DEFINE_boolean("profile", False, "Print profiling timers on completion.")


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

    linter_names = list(
        type(lin).__name__ for lin in self.dirlinters + self.filelinters)
    app.Log(2, "Running //%s linters: %s", self.toplevel_dir,
            ", ".join(linter_names))

  def _LintThisDirectory(
      self, abspath: str, relpath: str, dirnames: typing.List[str],
      filenames: typing.List[str]) -> typing.List[linters.Error]:
    """Run linters in this directory."""
    errors = []

    # Strip files and directories which are not to be linted.
    dirnames = [d for d in dirnames if d not in common.IGNORED_DIRS]
    filenames = [f for f in filenames if f not in common.IGNORED_FILES]

    for linter in self.dirlinters:
      errors += linter(abspath, relpath, dirnames, filenames)

    for filename in filenames:
      for linter in self.filelinters:
        errors += linter(f"{abspath}/{filename}", f"{relpath}/{filename}",
                         filename) or []

    return errors

  def __call__(self, *args, **kwargs):
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
  lintercache.LinterCache(workspace_)
  xmp_cache.XmpCache(workspace_)
  WorkspaceLinter(workspace_)()

  # Print the carriage return once we've done updating the counts line.
  if FLAGS.counts and linters.ERROR_COUNTS:
    print("", file=sys.stderr)

  # Print the profiling timers once we're done.
  if FLAGS.profile:
    total_time = TIMERS.total_seconds
    linting_time = TIMERS.linting_seconds
    cached_time = TIMERS.cached_seconds
    overhead = total_time - linting_time - cached_time

    print(
        f'linting={linting_time:.3f}s, cached={cached_time:.3f}s, '
        f'overhead={overhead:.3f}s, total={total_time:.3f}s',
        file=sys.stderr)


if __name__ == "__main__":
  try:
    app.RunWithArgs(main)
  except KeyboardInterrupt:
    print("interrupt")
    sys.exit(1)
