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
"""An opinionated, non-configurable enforcer of code style.

Usage:
  $ format <path ...>

This program enforces a consistent code style on files by modifying them in
place. If a path is a directory, all files inside it are formatted.

Features:

  * Consistent code styling of C/C++, Python, Java, SQL, JavaScript, HTML,
    Protocol Buffers, CSS, go, and JSON files.
  * A git-aware `--pre_commit` mode which formats changed files and signs off
    commits. Commites are rejected if a partially-staged file is modified.
    Enforce the use of pre-commit mode (and add a "Signed off" footer to
    commits) by running `--install_pre_commit_hook`.
  * Fast incremental formats of large code bases using a "last modified"
    time stamp cache.
  * Support for `.formatignore` files to mark files to be excluded from
    formatting. The syntax of ignore files is similar to `.gitignore`, e.g. a
    list of patterns to match, including (recursive) glob expansion, and
    patterns beginning with `!` are un-ignored.
  * Safe execution using inter-process locking to prevent multiple formatters
    modifying files simultaneously.

The type of formatting applied to a file is determined by its suffix. See
format --print_suffixes for a list of suffixes which are formatted.

This program uses a filesystem cache to store various attributes such as a
database of file modified times. See `format --print_cache_path` to print the
path of the cache. Included in the cache is a file lock which prevents mulitple
instances of this program from modifying files at the same time, irrespective
of the files being formatted.
"""
from labm8.py import app
from tools.format import app_paths
from tools.format import format_paths
from tools.format import path_generator as path_generators
from tools.format import pre_commit
from tools.format.default_suffix_mapping import (
  mapping as default_suffix_mapping,
)


FLAGS = app.FLAGS

app.DEFINE_boolean(
  "skip_git_submodules",
  True,
  "Check for, and exclude, git submodules from path expansion. This causes the "
  "formatter to respect git submodule boundaries, so that running format "
  "from within a git repository won't create dirty submodule states. You can "
  "still visit git submodules by naming them as arguments.",
)
app.DEFINE_boolean(
  "print_cache_path",
  False,
  "Print the path of the persistent filesystem cache and exit.",
)
app.DEFINE_boolean(
  "print_suffixes",
  False,
  "Print the list of filename suffixes which are formatted and return.",
)
app.DEFINE_boolean(
  "dry_run",
  False,
  "Only print the paths of files that will be formatted, without formatting "
  "them.",
)
app.DEFINE_boolean(
  "pre_commit",
  False,
  "Run formatter in pre-commit mode. When in pre-commit mode, all files that "
  "are staged for commit are formatted. If a formatter modifies a file that "
  "was staged for commit, the new changes are automatically staged. If a file "
  "was only partially staged for commit, then this program exits with a "
  "non-zero returncode, requiring you to review the formatter's changes "
  "before re-running the commit.",
)
app.DEFINE_boolean(
  "install_pre_commit_hook",
  False,
  "Install a pre-commit hook for the current git repository that runs this "
  "program.",
)


def Main(argv):
  """Main entry point."""
  if FLAGS.print_cache_path:
    print(app_paths.GetCacheDir())
    return
  elif FLAGS.print_suffixes:
    print("\n".join(sorted(default_suffix_mapping.keys())))
    return
  elif FLAGS.install_pre_commit_hook:
    pre_commit.InstallPreCommitHookOrDie()
    return

  path_generator = path_generators.PathGenerator(
    ".formatignore", skip_git_submodules=FLAGS.skip_git_submodules
  )

  args = argv[1:]

  # Resolve the paths to format.
  if FLAGS.pre_commit:
    if args:
      raise app.UsageError("--pre_commit takes no arguments")
    pre_commit.Main()
    return
  elif not args:
    raise app.UsageError("No paths were provided to format.")
  else:
    paths = path_generator.GeneratePaths(args)

  format_paths.FormatPathsOrDie(paths, dry_run=FLAGS.dry_run)


if __name__ == "__main__":
  app.RunWithArgs(Main)
