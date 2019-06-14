# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""This file defines the GitHubRepo class."""
import binascii
import collections
import hashlib
import multiprocessing
import pathlib
import subprocess
import typing

import progressbar

from datasets.github.scrape_repos.preprocessors import preprocessors
from datasets.github.scrape_repos.preprocessors import public
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import app
from labm8 import humanize
from labm8 import pbutil

FLAGS = app.FLAGS

IndexProgress = collections.namedtuple('IndexProgress', ['i', 'n'])


class GitHubRepo(object):
  """Representation of a GitHub repo."""

  def __init__(self, metafile: pathlib.Path):
    """Instantiate a github repo.

    Args:
      metafile: The path to the github meta file proto.

    Raises:
      ValueError: In case the metafile cannot be read.
    """
    self.metafile: pathlib.Path = metafile
    try:
      self.meta: scrape_repos_pb2.GitHubRepoMetadata = pbutil.FromFile(
          metafile, scrape_repos_pb2.GitHubRepoMetadata())
    except pbutil.DecodeError as e:
      raise ValueError(f"Failed to read metafile '{self.metafile}' {e}")
    self.name: str = f'{self.meta.owner}_{self.meta.name}'
    self.clone_dir: pathlib.Path = metafile.parent / self.name
    self.index_dir = (pathlib.Path(str(metafile.parent) + '.index') / self.name)

  def IsCloned(self) -> bool:
    """Return whether the repo has been cloned."""
    return (self.clone_dir / '.git').is_dir()

  def IsIndexed(self) -> bool:
    """Return whether the repo has been indexed."""
    return (self.index_dir / 'DONE.txt').is_file()

  def Clone(self) -> 'GitHubRepo':
    """Clone the repo."""
    if self.IsCloned():
      return self

    raise NotImplementedError

  def Index(self,
            indexers: typing.List[scrape_repos_pb2.ContentFilesImporterConfig],
            pool: multiprocessing.Pool,
            i: IndexProgress = None) -> 'GitHubRepo':
    """Index the repo."""
    if self.IsCloned() and not self.IsIndexed():
      self.index_dir.mkdir(parents=True, exist_ok=True)
      for indexer in indexers:
        self._IndexPattern(indexer, pool, i)
      (self.index_dir / 'DONE.txt').touch()
    return self

  def _IndexPattern(self, indexer: scrape_repos_pb2.ContentFilesImporterConfig,
                    pool: multiprocessing.Pool,
                    i: IndexProgress) -> 'GitHubRepo':
    """Index the repo."""
    pattern = indexer.source_code_pattern
    pattern = (f'{self.clone_dir}/{pattern[1:]}'
               if pattern[0] == '^' else f'{self.clone_dir}/{pattern}')
    cmd = [
        'find',
        str(self.clone_dir), '-type', 'f', '-regex', pattern, '-not', '-path',
        '*/.git/*'
    ]
    app.Log(2, '$ %s', ' '.join(cmd))
    paths = subprocess.check_output(
        cmd, universal_newlines=True).rstrip().split('\n')
    if len(paths) == 1 and not paths[0]:
      app.Log(2, 'No files to import from %s', self.clone_dir)
      return self
    if i:
      app.Log(1, "[%s / %s] Importing %s files from %s ...", i.i, i.n,
              humanize.Commas(len(paths)), self.name)
    else:
      app.Log(1, "Importing %s files from %s ...", humanize.Commas(len(paths)),
              self.name)
    all_files_relpaths = public.GetAllFilesRelativePaths(self.clone_dir)
    jobs = (scrape_repos_pb2.ImportWorker(
        clone_from_url=self.meta.clone_from_url,
        clone_dir=str(self.clone_dir),
        abspath=p,
        all_files_relpaths=all_files_relpaths,
        preprocessors=indexer.preprocessor,
        index_dir=str(self.index_dir),
    ) for p in paths)
    progress_bar = progressbar.ProgressBar(max_value=len(paths))
    for _ in progress_bar(pool.imap_unordered(IndexContentFiles, jobs)):
      pass

  def ContentFiles(self) -> typing.Iterable[scrape_repos_pb2.ContentFile]:
    """Return an iterator over all contentfiles in the repo."""
    if self.IsIndexed():
      return (pbutil.FromFile(f, scrape_repos_pb2.ContentFile())
              for f in self.index_dir.iterdir()
              if f.name != 'DONE.txt')
    else:
      return []


def IndexContentFiles(job: scrape_repos_pb2.ImportWorker) -> None:
  """Index content files."""
  relpath = job.abspath[len(str(job.clone_dir)) + 1:]
  try:
    texts = preprocessors.Preprocess(pathlib.Path(job.clone_dir), relpath,
                                     job.all_files_relpaths, job.preprocessors)
    for i, text in enumerate(texts):
      sha256 = hashlib.sha256(text.encode('utf-8'))
      proto = scrape_repos_pb2.ContentFile(clone_from_url=job.clone_from_url,
                                           relpath=relpath,
                                           artifact_index=i,
                                           sha256=sha256.digest(),
                                           charcount=len(text),
                                           linecount=len(text.split('\n')),
                                           text=text)
      path = pathlib.Path(job.index_dir) / (
          binascii.hexlify(proto.sha256).decode('utf-8') + '.pbtxt')
      pbutil.ToFile(proto, path)
  except UnicodeDecodeError:
    app.Warning('Failed to decode %s', relpath)
