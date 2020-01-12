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


def GetStagedPathsOrDie(
  path_generator,
) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
  os.chdir(GetGitRootOrDie())

  staged_paths, partially_staged_paths = [], []

  staged_relpaths = GetModifiedFilesOrDie(staged=True)
  unstaged_relpaths = GetModifiedFilesOrDie(staged=False)
  partially_staged_relpaths = list(
    set(staged_relpaths).intersection(set(unstaged_relpaths))
  )

  for relpath in staged_relpaths:
    paths = list(path_generator.GeneratePaths([relpath]))
    assert len(paths) <= 1
    if paths:
      staged_paths.append(paths[0])

  for relpath in partially_staged_relpaths:
    paths = list(path_generator.GeneratePaths([relpath]))
    assert len(paths) <= 1
    if paths:
      partially_staged_paths.append(paths[0])

  return staged_paths, partially_staged_paths


def GitAddOrDie(paths: Iterable[pathlib.Path]):
  try:
    subprocess.check_call(["git", "add"] + [str(x) for x in paths])
  except subprocess.CalledProcessError:
    print(f"ERROR: git add faled: {paths}", file=sys.stderr)
    sys.exit(1)


def InstallPreCommitHookOrDie():
  git_root = GetGitRootOrDie()

  hooks_dir = git_root / ".git" / "hooks"

  if not hooks_dir.is_dir():
    print(f"ERROR: git hooks directory not found: {hooks_dir}", file=sys.stderr)
    sys.exit(1)

  pre_commit = hooks_dir / "pre-commit"
  if pre_commit.is_file():
    os.unlink(pre_commit)

  with open(pre_commit, "w") as f:
    f.write(f"set -eu\n{sys.argv[0]} --pre_commit\n")
  os.chmod(pre_commit, 0o744)
  print(pre_commit)
