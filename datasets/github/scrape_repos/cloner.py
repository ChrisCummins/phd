# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Clone GitHub repositories.

This looks for repo meta files and clones any which have not been cloned.
"""
import multiprocessing
import pathlib
import random
import subprocess
import threading
import typing
from typing import Optional

import progressbar

from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8.py import app
from labm8.py import fs
from labm8.py import humanize
from labm8.py import pbutil
from tools.git import git_clone

FLAGS = app.FLAGS

app.DEFINE_string(
  "cloner_clone_list", None, "The path to a LanguageCloneList file."
)
app.DEFINE_integer(
  "repository_clone_timeout_minutes",
  30,
  "The maximum number of minutes to attempt to clone a "
  "repository before "
  "quitting and moving on to the next repository.",
)
app.DEFINE_integer(
  "num_cloner_threads", 4, "The number of cloner threads to spawn."
)


def GetCloneDir(metafile: pathlib.Path) -> Optional[pathlib.Path]:
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  if not meta.owner and meta.name:
    app.Error("Metafile missing owner and name fields %s", metafile)
    return
  return metafile.parent / f"{meta.owner}_{meta.name}"


def CloneFromMetafile(metafile: pathlib.Path) -> None:
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  clone_dir = GetCloneDir(metafile)
  if not clone_dir:
    app.Error("Failed to determine clone directory")
  app.Log(2, "%s", meta)
  if (clone_dir / ".git").is_dir():
    return

  # Remove anything left over from a previous attempt.
  subprocess.check_call(["rm", "-rf", str(clone_dir)])

  # Try to checkout the repository and submodules.
  try:
    git_clone.GitClone(
      meta.clone_from_url,
      clone_dir,
      shallow=True,
      recursive=True,
      timeout=FLAGS.repository_clone_timeout_minutes * 60,
    )
  except git_clone.RepoCloneFailed:
    # Remove anything left over from a previous attempt.
    subprocess.check_call(["rm", "-rf", str(clone_dir)])
    # Try again, but this time without cloning submodules.
    try:
      git_clone.GitClone(
        meta.clone_from_url,
        clone_dir,
        shallow=True,
        recursive=False,
        timeout=FLAGS.repository_clone_timeout_minutes * 60,
      )
    except git_clone.RepoCloneFailed:
      # Give up.
      app.Warning("\nClone failed %s:\n%s", meta.clone_from_url)
      # Remove anything left over.
      subprocess.check_call(["rm", "-rf", str(clone_dir)])


def IsRepoMetaFile(f: str):
  """Determine if a path is a GitHubRepoMetadata message."""
  return fs.isfile(f) and pbutil.ProtoIsReadable(
    f, scrape_repos_pb2.GitHubRepoMetadata()
  )


class AsyncWorker(threading.Thread):
  """Thread which clones github repos."""

  def __init__(self, meta_files: typing.List[pathlib.Path]):
    super(AsyncWorker, self).__init__()
    self.meta_files = meta_files
    self.max = len(meta_files)
    self.i = 0

  def run(self):
    pool = multiprocessing.Pool(FLAGS.num_cloner_threads)
    for _ in pool.imap_unordered(CloneFromMetafile, self.meta_files):
      self.i += 1


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  clone_list_path = pathlib.Path(FLAGS.cloner_clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError("--clone_list is not a file.")
  clone_list = pbutil.FromFile(
    clone_list_path, scrape_repos_pb2.LanguageCloneList()
  )

  meta_files = []
  for language in clone_list.language:
    directory = pathlib.Path(language.destination_directory)
    if directory.is_dir():
      meta_files += [
        pathlib.Path(directory / f)
        for f in directory.iterdir()
        if IsRepoMetaFile(f)
      ]
  random.shuffle(meta_files)
  worker = AsyncWorker(meta_files)
  app.Log(1, "Cloning %s repos from GitHub ...", humanize.Commas(worker.max))
  bar = progressbar.ProgressBar(max_value=worker.max, redirect_stderr=True)
  worker.start()
  while worker.is_alive():
    bar.update(worker.i)
    worker.join(0.5)
  bar.update(worker.i)


if __name__ == "__main__":
  app.RunWithArgs(main)
