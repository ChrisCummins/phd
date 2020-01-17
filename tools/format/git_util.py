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
"""Python wrappers for interfacing with git."""
import os
import pathlib
import subprocess
import sys
from typing import Iterable
from typing import List
from typing import Tuple

from labm8.py import app

FLAGS = app.FLAGS


def GetGitRootOrDie() -> pathlib.Path:
  """Get the root directory of the current git repository.

  Returns:
    The absolute path of the git repository of the current working directory.
  """
  try:
    top_level = subprocess.check_output(
      ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    ).rstrip()
    return pathlib.Path(top_level)
  except subprocess.CalledProcessError:
    print("ERROR: Unable to get git directory root")
    sys.exit(1)


def GetModifiedFilesOrDie(staged: bool) -> List[str]:
  """List *either* the staged or unstaged files (not both).

  To list both, call this function twice with both staged=True and staged=False.

  Args:
    staged: Return staged files if True, else unstaged files.

  Returns:
    A list of paths relative to the repo root.
  """
  cmd = ["git", "diff", "--name-only"]
  if staged:
    cmd.append("--cached")
  try:
    output = subprocess.check_output(cmd, universal_newlines=True)
  except subprocess.CalledProcessError as e:
    print(f"ERROR: command failed: {' '.join(cmd)}", file=sys.stderr)
    sys.exit(1)

  lines = output.split("\n")
  staged_file_relpaths = lines[:-1]  # Last line is blank.
  return staged_file_relpaths


def GetStagedPathsOrDie(
  path_generator,
) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
  """

  Args:
    path_generator:

  Returns:

  """
  staged_paths, partially_staged_paths = [], []

  staged_relpaths = GetModifiedFilesOrDie(staged=True)
  unstaged_relpaths = GetModifiedFilesOrDie(staged=False)
  partially_staged_relpaths = list(
    set(staged_relpaths).intersection(set(unstaged_relpaths))
  )

  for relpath in staged_relpaths:
    paths = list(path_generator.GeneratePaths([relpath]))
    # A staged path can resolve to multiple real paths if it is a submodule
    # that is being altered. Ignore those.
    if len(paths) > 1:
      continue
    if paths:
      staged_paths.append(paths[0])

  for relpath in partially_staged_relpaths:
    paths = list(path_generator.GeneratePaths([relpath]))
    # A staged path can resolve to multiple real paths if it is a submodule
    # that is being altered. Ignore those.
    if len(paths) > 1:
      continue
    if paths:
      partially_staged_paths.append(paths[0])

  return staged_paths, partially_staged_paths


def GitAddOrDie(paths: Iterable[pathlib.Path]):
  try:
    subprocess.check_call(["git", "add"] + [str(x) for x in paths])
  except subprocess.CalledProcessError:
    print(f"ERROR: git add faled: {paths}", file=sys.stderr)
    sys.exit(1)
