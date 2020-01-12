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
"""Utilities for interfacing with git."""
import pathlib
import subprocess
import sys
from typing import List

from labm8.py import app

FLAGS = app.FLAGS


def GetGitRootOrDie() -> pathlib.Path:
  try:
    top_level = subprocess.check_output(
      ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    ).rstrip()
    return pathlib.Path(top_level)
  except subprocess.CalledProcessError:
    print("ERROR: Unable to get git directory root")
    sys.exit(1)


def GetModifiedFilesOrDie(staged: bool) -> List[pathlib.Path]:
  """List *either* the staged or unstaged files (not both).

  To list both, call this function twice with both staged=True and staged=False.
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
