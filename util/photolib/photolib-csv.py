"""A linter for ensuring that a Photo Library is organized correctly."""
import os
import sys

import pathlib
import typing

from labm8 import app
from util.photolib import linters
from util.photolib import workspace

FLAGS = app.FLAGS


class CsvDump(linters.ToplevelLinter):
  """A linter which dumps the XMP metadata of files as a CSV."""

  def __init__(self, *args, **kwargs):
    super(CsvDump, self).__init__(*args, **kwargs)

  def __call__(self, directory: pathlib.Path):
    """Run the linters."""
    if directory == self.workspace.workspace_root:
      directory = self.toplevel_dir

    directory_str = str(directory.absolute())
    toplevel_str = str(self.toplevel_dir.absolute())

    # Start at the top level.
    if not directory_str.startswith(toplevel_str):
      return

    for abspath, dirnames, filenames in os.walk(directory):
      relpath = self.workspace.GetRelpath(abspath)
      print("\033[KScanning", relpath, end=" ...\r")
      sys.stdout.flush()
      self._LintThisDirectory(abspath, relpath, dirnames, filenames)


def DumpCsvForDirsOrDie(paths: typing.List[pathlib.Path]) -> None:
  for path in paths:
    if not path.exists():
      app.FatalWithoutStackTrace(f"File or directory not found: '{path}'")

  # Linting is on a per-directory level, not per-file.
  directories_to_lint = {
      path if path.is_dir() else path.parent for path in paths
  }

  for i, directory in enumerate(sorted(directories_to_lint)):
    directory = directory.absolute()
    app.Log(2, 'Dump directory `%s`', directory)
    workspace_ = workspace.Workspace.FindWorkspace(directory)
    linter = CsvDump(workspace_, "photos", linters.CsvDirLinter,
                     linters.CsvFileLinter, linters.TIMERS)

    if not i:
      csv_dump = linters.CsvFileDumper(workspace_)
      csv_dump.csv_writer.writeheader()
    linter(directory)


def main(argv):
  """Main entry point."""
  dirs = [pathlib.Path(d) for d in (argv[1:] or [os.getcwd()])]
  DumpCsvForDirsOrDie(dirs)


if __name__ == "__main__":
  app.RunWithArgs(main)
