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
"""This module defines the inotify file watcher loop."""
import datetime
import pathlib
from typing import List

import inotify.adapters
import inotify.constants

from labm8.py import app
from tools.format import format_paths
from tools.format import path_generator as path_generators


FLAGS = app.FLAGS


def Format(paths: List[pathlib.Path]):
  """Run the formatter over a list of paths and print their outcomes."""
  # In the future I may want to refactor the FormatPaths class so that it can
  # process multiple "runs" rather than having to create and dispose of a
  # formatter each time we get a new FS event.
  for event in format_paths.FormatPaths(paths):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    prefix = f"[format {timestamp}]"
    if isinstance(event, Exception):
      print(prefix, "ERROR:", event)
    else:
      print(prefix, event)


def Main(args: List[str]):
  """Enter an infinite loop of watching and formatting files on change.

  !!!WARNING!!! This function never returns! The only way to break out of the
  infinite loop is via a keyboard interrupt signal.

  Args:
    args: A list of files or directories to watch for format events on.
  """
  if not args:
    raise app.UsageError("No paths to watch.")

  # inotify.adapters.InotifyTrees() doesn't seem to work for me, so I'm unable
  # to track new files created using this loop.

  # Expand all arguments into paths at launch time. This means that for
  # directory arguments, I am not able to get inotify notifications on new files
  # created.
  path_generator = path_generators.PathGenerator(
    ".formatignore", skip_git_submodules=FLAGS.skip_git_submodules
  )
  paths = list(path_generator.GeneratePaths(args))

  # Run an initial pass of the formatter before switching over to the
  # event-driven loop.
  Format(paths)

  # Register the inotify watchers for all paths.
  watchers = inotify.adapters.Inotify()
  for path in paths:
    watchers.add_watch(str(path), mask=inotify.constants.IN_MODIFY)

  # React to inotify events.
  try:
    for _, _, path, _ in watchers.event_gen(yield_nones=False):
      app.Log(2, "UPDATE %s", path)
      Format([pathlib.Path(path)])
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.RunWithArgs(Main)
