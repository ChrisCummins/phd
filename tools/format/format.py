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

  * Automated code styling of C/C++, Python, Java, SQL, JavaScript, HTML,
    CSS, go, and JSON files.
  * Support for `.formatignore` files to mark files to be excluded from
    formatting. The syntax of ignore files is similar to `.gitignore`, e.g. a
    list of patterns to match, including (recursive) glob expansion, and
    patterns beginning with `!` are un-ignored.
  * Persistent caching of "last modified" timestamps for files to minimize the
    amount of work done.
  * A process lock which prevents races when multiple formatters are launched
    simultaneously.

The type of formatting applied to a file is determined by its suffix. See
format --print_suffixes for a list of suffixes which are formatted.

This program uses a filesystem cache to store various attributes such as a
database of file modified times. See `format --print_cache_path` to print the
path of the cache. Included in the cache is a file lock which prevents mulitple
instances of this program from modifying files at the same time, irrespective
of the files being formatted.
"""
import os
import pathlib
import queue
import sys
from typing import Iterable
from typing import List

import appdirs
import fasteners

import build_info
from labm8.py import app
from tools.format import formatter_executor
from tools.format import git_util
from tools.format import path_generator as path_generators
from tools.format.formatters.suffix_mapping import mapping as formatters


FLAGS = app.FLAGS

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
app.DEFINE_boolean("pre_commit", False, "Run formatter as a pre-commit hook.")
app.DEFINE_boolean(
  "with_cache",
  True,
  'Enable the persistent caching of "last modified" timestamps for files. '
  "Files which have not changed since the last time the formatter was run are "
  "skipped. Running the formatter with --nowith_cache forces all files to be "
  "formatted, even if they have not changed.",
)


def GetCacheDir() -> pathlib.Path:
  """Resolve the cache directory for linters."""
  _BAZEL_TEST_TMPDIR = os.environ.get("TEST_TMPDIR")
  if _BAZEL_TEST_TMPDIR:
    os.environ["XDG_CACHE_HOME"] = _BAZEL_TEST_TMPDIR
  return pathlib.Path(
    appdirs.user_cache_dir(
      "phd_format", "Chris Cummins", version=build_info.GetBuildInfo().version
    )
  )


def Main(argv):
  cache_dir = GetCacheDir()
  cache_dir.mkdir(parents=True, exist_ok=True)

  if FLAGS.print_cache_path:
    print(cache_dir)
    return
  elif FLAGS.print_suffixes:
    print("\n".join(sorted(formatters.keys())))
    return

  # Acquire an inter-process lock. This does not need to be released - cleanup
  # of inter-process locks using the fasteners library is automatic. This will
  # block indefinitely if the lock is already acquired by a different process,
  # ensuring that only a single formatter is running at a time.
  lock_file = cache_dir / "LOCK"
  app.Log(3, "Acquiring lock file %s", lock_file)
  assert fasteners.InterProcessLock(lock_file)
  app.Log(3, "Lock file acquired")

  args = argv[1:]
  path_generator = path_generators.PathGenerator(".formatignore")

  # Resolve the paths to format.
  if FLAGS.pre_commit:
    if args:
      raise app.UsageError("--pre_commit takes no arguments")

    # In --pre_commit mode, we take the union of staged and partially-staged
    # files to format.
    staged_paths, partially_staged_paths = git_util.GetStagedPathsOrDie(
      path_generator
    )

    paths = set(staged_paths).union(set(partially_staged_paths))
  elif not args:
    raise app.UsageError("Must provide a path")
  else:
    paths = path_generator.GeneratePaths(args)

  executor = FormatPathsOrDie(cache_dir, paths)

  if FLAGS.pre_commit and executor.modified_files:
    # When in --pre_commit mode, a non-zero status means that staged files were
    # modified.
    modified_files = set(executor.modified_files)

    need_review = modified_files.intersection(set(partially_staged_paths))
    to_commit = modified_files - need_review

    if to_commit:
      print("✅  Modified files that will be automatically committed:")
      for path in sorted(to_commit):
        print("   ", path)
      git_util.GitAddOrDie(to_commit)

    if need_review:
      print(
        "⚠️  Partially staged modified files that must be inspected:",
        file=sys.stderr,
      )

      for path in sorted(need_review):
        print("   ", path, file=sys.stderr)
      print("[action] Selectively add unstaged changes using:", file=sys.stderr)
      print("    git add --patch --", *need_review, file=sys.stderr)
      sys.exit(1)


def FormatPathsOrDie(cache_dir: pathlib.Path, paths: List[pathlib.Path]):
  """Run the formatter on a list of arguments.

  Returns:
    The formatter executor.
  """
  q = queue.Queue()
  executor = formatter_executor.FormatterExecutor(cache_dir, q)

  # --dry_run flag to print the paths that would be formatted.
  if FLAGS.dry_run:
    for path in paths:
      print(path)
    return executor

  executor.start()

  for path in paths:
    # Check if there are corresponding formatters, and if so, send it off to
    # the executor to process.
    key = path.suffix or path.name
    if key in formatters:
      q.put(path)

  q.put(None)
  executor.join()

  if executor.errors:
    sys.exit(2)

  return executor


if __name__ == "__main__":
  app.RunWithArgs(Main)
