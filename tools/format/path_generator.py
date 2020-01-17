# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module generates filesystem paths from program arguments."""
import glob
import os
import pathlib
from typing import Iterable
from typing import List
from typing import Set

from labm8.py import app

FLAGS = app.FLAGS


class PathGenerator(object):
  """A a class for generating filesystem paths from program arguments.

  See GeneratePaths() for usage.
  """

  def __init__(self, ignore_file_name: str, skip_git_submodules: bool = True):
    self.ignore_file_name = ignore_file_name
    self.skip_git_submodules = skip_git_submodules
    self.ignored_paths: Set[pathlib.Path] = set()
    self.visited_ignore_files: Set[pathlib.Path] = set()

  def GeneratePaths(self, args: List[str]) -> Iterable[pathlib.Path]:
    """Enumerate file paths from a list of arguments.

    For each arg:
      1. Expand any globs using UNIX glob expansion.
      2. If the path is a directory, enumerate all files inside the directory
         and any subdirectories.
      3. Resolve the absolute path of any files.
      4. Check the contents of 'ignore files' to see if the path should be
         excluded. See VisitIgnoreFile() for details.

    Args:
      args: A list of arguments.

    Returns:
      An iterator over absolute pathlib.Path instances. Every path returned
      is a unique file that exists
    """
    visited_paths = set()

    for arg in args:
      arg_path = pathlib.Path(arg).absolute()

      # Sorting the result of globbing here rather than using the more efficient
      # glob.iglob() gives us a stable and expected iteration order, but has a
      # memory overhead as we must fully expand the glob before iterating
      # through the results.
      for path in sorted(glob.glob(arg, recursive=True)):
        path = pathlib.Path(path).absolute()

        if path.is_dir():
          # Iterate over the contents of directory arguments.
          for root, dirs, files in os.walk(path):
            root = pathlib.Path(root).absolute()

            if self.skip_git_submodules:
              # Don't visit a git submodule unless it was explicitly requested
              # as as argument. This prevents glob expansion from descending
              # into submodules.
              if root != arg_path and (root / ".git").is_file():
                break
              # Don't descend into git submodules.
              dirs[:] = [d for d in dirs if not (root / d / ".git").is_file()]

            # Only iterate through the directory contents if the directory is
            # not ignored.
            if not self.IsIgnored(root):
              # As with the glob expansion above, we sort the order of files
              # when iterating through directories so that we have a sensible
              # and stable iteration order. This has a performance hit for
              # very large directories.
              for file in sorted(files):
                path = root / file
                if path not in visited_paths and not self.IsIgnored(path):
                  visited_paths.add(path)
                  yield path
        else:
          if path not in visited_paths and not self.IsIgnored(path):
            visited_paths.add(path)
            yield path

  def IsIgnored(self, path: pathlib.Path) -> bool:
    """Determine if the path is ignored.

    Do this by visiting all "ignore files" on the filesystem path, starting with
    the current directory and working up to the filesystem root.

    Args:
      path: An absolute path.

    Returns:
      True if the path should be ignored, else False.
    """
    # Never descend into git directories.
    if ".git" in path.parts:
      return True

    for parent in path.parents:
      ignore_file = parent / self.ignore_file_name
      if ignore_file.is_file() and ignore_file not in self.visited_ignore_files:
        self.VisitIgnoreFile(ignore_file)
        self.visited_ignore_files.add(ignore_file)

    if path in self.ignored_paths:
      return True

    for parent in path.parents:
      if parent in self.ignored_paths:
        return True

    return False

  def VisitIgnoreFile(self, ignore_file: pathlib.Path) -> None:
    """Visit an ignore file and expand patterns in it.

    An ignore file is a list of patterns for files to exclude. The syntax of
    files emulates the .gitignore format. For example:

        # This is an ignore file. "#" is a comment character.
        hello.txt  # Patterns which match files are excluded.
        docs  # If a pattern matches a directory,
        **/*.o  # Globs are expanded, including recursively.
        !important.o  # Lines beggining with '!' are un-ignored.

    Args:
      ignore_file: The path of an ignore file.
    """
    app.Log(4, "visting ignore file %s", ignore_file)
    with open(ignore_file) as f:
      for line in f:
        components = line.split("#")
        pattern = components[0].strip()
        if pattern and pattern[0] == "!":
          # Un-ignore patterns, if they were previously marked as ignored.
          for path in glob.iglob(
            str(ignore_file.parent / pattern[1:]), recursive=True
          ):
            path = pathlib.Path(path)
            if path in self.ignored_paths:
              self.ignored_paths.remove(path)
        elif pattern:
          # Ignore patterns.
          for path in glob.iglob(
            str(ignore_file.parent / pattern), recursive=True
          ):
            self.ignored_paths.add(pathlib.Path(path))
