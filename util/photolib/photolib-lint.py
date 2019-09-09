"""A linter for ensuring that a Photo Library is organized correctly."""
import os
import sys

import pathlib
import typing

from labm8 import app
from labm8 import humanize
from util.photolib import linters
from util.photolib import workspace

FLAGS = app.FLAGS
app.DEFINE_boolean("profile", True, "Print profiling timers on completion.")
app.DEFINE_boolean("rm_errors_cache", False,
                   "If true, empty the errors cache prior to running.")


def LintPathsOrDie(paths: typing.List[pathlib.Path]) -> None:
  for path in paths:
    if not path.exists():
      app.FatalWithoutStackTrace(f"File or directory not found: '{path}'")

  # Linting is on a per-directory level, not per-file.
  directories_to_lint = {
      path if path.is_dir() else path.parent for path in paths
  }

  for directory in sorted(directories_to_lint):
    directory = directory.absolute()
    app.Log(2, 'Linting directory `%s`', directory)
    workspace_ = workspace.Workspace.FindWorkspace(directory)
    linters.Lint(workspace_, directory)

  # Print the carriage return once we've done updating the counts line.
  if FLAGS.counts:
    if linters.ERROR_COUNTS:
      print("", file=sys.stderr)
  else:
    linters.PrintErrorCounts(end="\n")

  # Print the profiling timers once we're done.
  if FLAGS.profile:
    total_time = linters.TIMERS.total_seconds
    linting_time = linters.TIMERS.linting_seconds
    cached_time = linters.TIMERS.cached_seconds
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


def main(argv):
  """Main entry point."""
  dirs = [pathlib.Path(d) for d in (argv[1:] or [os.getcwd()])]
  LintPathsOrDie(dirs)


if __name__ == "__main__":
  app.RunWithArgs(main)
