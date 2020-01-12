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
"""This module implement pre-commit behavior for formatter."""
import os
import sys

from labm8.py import app
from labm8.py import crypto
from tools.format import git_util


FLAGS = app.FLAGS


def GetArgsFromGitDiff():
  os.chdir(git_util.GetGitRootOrDie())

  staged_files = [
    f for f in git_util.GetModifiedFilesOrDie(staged=True) if os.path.isfile(f)
  ]
  unstaged_files = git_util.GetModifiedFilesOrDie(staged=False)

  # files_that_exist = [f for f in staged_files if os.path.isfile(f)]

  # A list of files that will be automatically staged for commit.
  files_to_stage = []
  # A list of files that were partially staged and have been modified by this
  # script. These must be hand-inspected and will cause the commit to abort.
  partially_staged_modified_files = []

  # Files that are both staged and unstaged are partially staged.
  partially_staged_files = list(
    set(staged_files).intersection(set(unstaged_files))
  )
  # We're only interest in files that exist.
  partially_staged_files = [
    f for f in partially_staged_files if os.path.isfile(f)
  ]

  # partially_staged_checksums = [
  #   crypto.md5_file(path) for path in partially_staged_files
  # ]

  # TODO: Run gazelle before linters.
  # Gazelle.MaybeRunGazelle(files_that_exist)
  #
  # # Check to see if gazelle modified any partially-staged files.
  # new_partially_staged_checksums = GetMd5sumOutputAsString(
  #     partially_staged_files
  # )
  # if new_partially_staged_checksums != partially_staged_checksums:
  #   # Go line by line through the md5sum outputs to find the differing files.
  #   for left, right in zip(
  #       partially_staged_checksums.split("\n"),
  #       new_partially_staged_checksums.split("\n"),
  #   ):
  #     if left != right:
  #       partially_staged_modified_files.append(" ".join(left.split()[1:]))

  app.Log(1, "STAGED %s", staged_files)
  app.Log(1, "PARTIALLY STAGED %s", partially_staged_files)
  app.Log(1, "unstaged %s", unstaged_files)

  return set(staged_files).union(set(partially_staged_files))
  #
  # linters = git_util.LinterActions(files_that_exist)
  # num_actions = len(linters.paths_with_actions)
  #
  # if num_actions:
  #   task_start_time = time.time()
  #   git_util.Print("Running", num_actions, "linter actions ...", end=" ")
  #   linters.RunOrDie()
  #   git_util.Print("ok  {:.3f}s".format(time.time() - task_start_time))
  #
  #   # Get a list of partially-staged files that were modified by the linters.
  #   partially_staged_files = set(
  #       f for f in linters.modified_paths if f in unstaged_files
  #   )
  #   partially_staged_modified_files += list(partially_staged_files)
  #
  #   # For every file in the git index where the entire diff was indexed,
  #   # re-index it.
  #   files_to_stage += [
  #     f for f in linters.modified_paths if f not in partially_staged_files
  #   ]
  #
  #   if files_to_stage:
  #     files_to_stage = list(set(files_to_stage))
  #     git_util.Print("Modified files that will be automatically committed:")
  #     for path in sorted(files_to_stage):
  #       git_util.Print("   ", path)
  #     GitAddOrDie(files_to_stage)
  #
  #   # Partially-staged files (i.e. the result of running `git add -p` cannot
  #   # be simply re-indexed and committed since we don't know what was modified
  #   # by the linter and what was deliberately left unstaged.
  #   if partially_staged_modified_files:
  #     partially_staged_modified_files = list(
  #         sorted(set(partially_staged_modified_files))
  #     )
  #     git_util.Print(
  #         "⚠️  Partially staged modified files that must be inspected:"
  #     )
  #     for path in partially_staged_modified_files:
  #       git_util.Print("   ", path)
  #     git_util.Print()
  #     git_util.Print("[action] Selectively add unstaged changes using:")
  #     git_util.Print()
  #     git_util.Print(
  #         "    git add --patch --", *partially_staged_modified_files
  #     )
  #     git_util.Print()
  #     sys.exit(1)
  #
  #   git_util.Print(
  #       "✅  Pre-commit checks passed in {:.3f}s".format(time.time() - start_time)
  #   )
