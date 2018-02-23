"""A linter for ensuring that a Photo Library is organized correctly."""
import os
import sys
import typing

from absl import app
from absl import flags
from absl import logging

from photolib import linters
from photolib import util
from photolib import workspace

FLAGS = flags.FLAGS
flags.DEFINE_string("workspace", os.getcwd(), "Path to workspace root")


class ToplevelLinter(linters.Linter):
  """A linter for top level directories."""
  __cost__ = 1

  def __init__(self, workspace_abspath: str, toplevel_dir: str,
               dirlinters: typing.List[linters.DirLinter],
               filelinters: typing.List[linters.FileLinter]):
    super(ToplevelLinter, self).__init__()
    self.workspace = workspace_abspath
    self.toplevel_dir = toplevel_dir
    self.dirlinters = linters.get_linters(dirlinters)
    self.filelinters = linters.get_linters(filelinters)

    linter_names = [
        type(lin).__name__ for lin in self.dirlinters + self.filelinters]
    logging.info("Running //%s linters: %s",
                 self.toplevel_dir, ", ".join(linter_names))

  def __call__(self, *args, **kwargs):
    working_dir = os.path.join(self.workspace, self.toplevel_dir)
    for abspath, dirnames, filenames in os.walk(working_dir):
      relpath = workspace.get_workspace_relpath(self.workspace, abspath)

      logging.debug("traversing %s", relpath)

      # Strip files and directories which are not to be linted.
      dirnames = [d for d in dirnames if d not in util.IGNORED_DIRS]
      filenames = [f for f in filenames if f not in util.IGNORED_FILES]

      for linter in self.dirlinters:
        linter(abspath, relpath, dirnames, filenames)

      for filename in filenames:
        for linter in self.filelinters:
          linter(f"{abspath}/{filename}", f"{relpath}/{filename}",
                 filename)


class WorkspaceLinter(linters.Linter):
  """The master linter for the photolib workspace."""
  __cost__ = 1

  def __init__(self, abspath: str):
    super(WorkspaceLinter, self).__init__()
    self.workspace = abspath

  def __call__(self, *args, **kwargs):
    photolib_linter = ToplevelLinter(
        self.workspace, "photos",
        linters.PhotolibDirLinter, linters.PhotolibFileLinter)
    gallery_linter = ToplevelLinter(
        self.workspace, "gallery",
        linters.GalleryDirLinter, linters.GalleryFileLinter)

    photolib_linter()
    gallery_linter()


def main(argv):  # pylint: disable=missing-docstring
  del argv
  abspath = workspace.find_workspace_rootpath(
      os.path.expanduser(FLAGS.workspace))
  if not abspath:
    print(f"Cannot find workspace in '{FLAGS.workspace}'", file=sys.stderr)
    sys.exit(1)

  WorkspaceLinter(abspath)()

  # Print the carriage return once we've done updating the counts line.
  if FLAGS.counts and linters.ERROR_COUNTS:
    print("", file=sys.stderr)


if __name__ == "__main__":
  try:
    app.run(main)
  except KeyboardInterrupt:
    print("interrupt")
    sys.exit(1)
