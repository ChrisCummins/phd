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
"""This module defines the git pre-commit mode behavior for format."""
import os
import sys
from typing import Iterable

import build_info
from labm8.py import app
from labm8.py import fs
from tools.format import app_paths
from tools.format import format_paths
from tools.format import git_util
from tools.format import path_generator as path_generators


FLAGS = app.FLAGS


def Main():
  """Run the pre-commit mode formatter."""
  # Get and change into the root directory of this repository.
  git_root = git_util.GetGitRootOrDie()
  os.chdir(git_root)

  path_generator = path_generators.PathGenerator(
    ".formatignore", skip_git_submodules=FLAGS.skip_git_submodules
  )

  # In --pre_commit mode, we take the union of staged and partially-staged
  # files to format.
  staged_paths, partially_staged_paths = git_util.GetStagedPathsOrDie(
    path_generator
  )

  paths = set(staged_paths).union(set(partially_staged_paths))

  modified_paths = format_paths.FormatPathsOrDie(paths)

  modified_paths = set(modified_paths)

  # Write the list of changed files that is read by the git commit message hook,
  # if used.
  WriteModifiedFilesListForCommitMessage(
    sorted([os.path.relpath(path, git_root) for path in modified_paths])
  )

  if modified_paths:
    need_review = modified_paths.intersection(set(partially_staged_paths))
    to_commit = modified_paths - need_review

    if to_commit:
      print("✅  Modified files that will be automatically committed:")
      for path in sorted(to_commit):
        print("   ", os.path.relpath(path, git_root))
      git_util.GitAddOrDie(to_commit)

    if need_review:
      print(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        file=sys.stderr,
      )
      print(
        "⚠️  Partially staged modified files that must be inspected:",
        file=sys.stderr,
      )
      for path in sorted(need_review):
        print("   ", os.path.relpath(path, git_root), file=sys.stderr)

      print("", file=sys.stderr)
      print("[action] Selectively add unstaged changes using:", file=sys.stderr)
      print("    git add --patch --", *need_review, file=sys.stderr)
      sys.exit(3)


def WriteModifiedFilesListForCommitMessage(modified_files: Iterable[str]):
  """Write a list of modified files which will be added to git commit message.

  Args:
    A list of paths modified by format.
  """
  s = [""]
  if modified_files:
    s.append("# `format --pre_commit` modified the following files:")
    for path in modified_files:
      s.append(f"#    {path}")
  else:
    s.append("# `format --pre_commit` made no changes.")
  s.append("\n")
  fs.Write(
    app_paths.GetCacheDir() / "commit_message_changelist.txt",
    "\n".join(s).encode("utf-8"),
  )


def InstallPreCommitHookOrDie():
  git_root = git_util.GetGitRootOrDie()

  hooks_dir = git_root / ".git" / "hooks"

  if not hooks_dir.is_dir():
    print(f"ERROR: git hooks directory not found: {hooks_dir}", file=sys.stderr)
    sys.exit(1)

  # Install pre-commit hook which runs this program in --pre_commit mode.
  pre_commit = hooks_dir / "pre-commit"
  if pre_commit.is_file():
    os.unlink(pre_commit)

  with open(pre_commit, "w") as f:
    f.write(f"#!/usr/bin/env bash\nset -e\nformat --pre_commit\n")
  os.chmod(pre_commit, 0o744)
  print(pre_commit)

  # Install prepare-commit-msg hook which adds a "signed off" hook.
  prepare_commit_msg = hooks_dir / "prepare-commit-msg"
  if prepare_commit_msg.is_file():
    os.unlink(prepare_commit_msg)

  version = build_info.GetBuildInfo().version
  with open(prepare_commit_msg, "w") as f:
    commit_message_changelist = (
      app_paths.GetCacheDir() / "commit_message_changelist.txt"
    )
    tmp_commit_message_path = app_paths.GetCacheDir() / "git_commit_message.txt"
    f.write(
      f"""\
#!/usr/bin/env bash
set -e
if [[ -z "$2" ]]; then
  cp "$1" "{tmp_commit_message_path}"
  echo -e "\\n\\nSigned-off-by: format {version} <github.com/ChrisCummins/format>" > "$1"
  cat "{commit_message_changelist}" "{tmp_commit_message_path}" >> "$1"
  rm -f "{tmp_commit_message_path}"
fi
"""
    )
  os.chmod(prepare_commit_msg, 0o744)
  print(prepare_commit_msg)


if __name__ == "__main__":
  app.Run(Main)
