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
"""This module converts program arguments into a list of paths."""
import glob
import os
import pathlib
from typing import List

from labm8.py import app


FLAGS = app.FLAGS


def GeneratePaths(args: List[str]):
  """Enumerate the paths from a list of args.

  For each arg:
    1. Expand any globs using UNIX glob expansion.
    2. If the path is a directory, enumerate all files inside the directory
       and any subdirectories.
  """
  # TODO: Look for .formatignore files and ignore paths from them.
  for arg in args:
    for path in glob.iglob(arg):
      path = pathlib.Path(path)
      if path.is_dir():
        for root, dirs, files in os.walk(path):
          for file in files:
            yield pathlib.Path(root) / file
      else:
        yield path
