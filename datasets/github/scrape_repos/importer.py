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
"""Import files into a ContentFiles database."""
import multiprocessing
import os
from sqlalchemy import orm

import hashlib
import pathlib
import progressbar
import random
import subprocess
import typing
from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import app
from labm8 import humanize
from labm8 import pbutil

from datasets.github.scrape_repos.preprocessors import preprocessors
from datasets.github.scrape_repos.preprocessors import public

FLAGS = app.FLAGS
app.DEFINE_integer('processes', os.cpu_count(),
                   'The number of simultaneous processes.')

app.DEFINE_string('importer_clone_list', None,
                  'The path to a LanguageCloneList file.')


def ShouldImportRepo(session: orm.session.Session,
                     metafile: pathlib.Path) -> bool:
  """Determine if the repository described by a metafile should be imported.

  A repository should be imported iff:
    * The metafile is a valid GitHubRepoMetadata proto.
    * The clone directory specified in the metafile appears to be a github repo.
    * The repo does not exist in the contentfiles database.
  """
  if not (metafile.is_file() and pbutil.ProtoIsReadable(
      metafile, scrape_repos_pb2.GitHubRepoMetadata())):
    return False
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  clone_dir = metafile.parent / f'{meta.owner}_{meta.name}'
  if not (clone_dir / '.git').is_dir():
    return False
  return not contentfiles.GitHubRepository.IsInDatabase(session, meta)


def ImportWorker(job: scrape_repos_pb2.ImportWorker
                ) -> typing.List[contentfiles.ContentFile]:
  """Import a content file."""
  relpath = job.abspath[len(str(job.clone_dir)) + 1:]
  outputs: typing.List[contentfiles.ContentFile] = []
  try:
    texts = preprocessors.Preprocess(
        pathlib.Path(job.clone_dir), relpath, job.all_files_relpaths,
        job.preprocessors)
    for i, text in enumerate(texts):
      encoded_text = text.encode('ascii', 'ignore')
      sha256 = hashlib.sha256(encoded_text).hexdigest()
      text = encoded_text.decode('ascii')
      outputs.append(
          contentfiles.ContentFile(
              clone_from_url=job.clone_from_url,
              relpath=relpath,
              artifact_index=i,
              sha256=sha256,
              charcount=len(text),
              linecount=len(text.split('\n')),
              text=text))
  except UnicodeDecodeError:
    app.Warning('Failed to decode %s', relpath)
  return outputs


def ImportRepo(session: orm.session.Session,
               language: scrape_repos_pb2.LanguageToClone,
               metafile: pathlib.Path, pool: multiprocessing.Pool) -> None:
  """Import contentfiles from repository.

  Args:
    session: A database session to import to.
    language: The language specification for the repo.
    metafile: The repo metafile.
    pool: A multiprocessing pool.
  """
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  clone_dir = metafile.parent / f'{meta.owner}_{meta.name}'
  repo = contentfiles.GitHubRepository.GetOrAdd(session, meta)
  repo.language = language.language
  session.flush()

  for importer in language.importer:
    if not importer.source_code_pattern:
      app.Error('No source_code_pattern specified! Stopping now.')
      return

    pat = importer.source_code_pattern
    pat = f'{clone_dir}/{pat[1:]}' if pat[0] == '^' else f'{clone_dir}/{pat}'
    cmd = [
        'find',
        str(clone_dir), '-type', 'f', '-regex', pat, '-not', '-path', '*/.git/*'
    ]
    app.Log(2, '$ %s', ' '.join(cmd))
    paths = subprocess.check_output(
        cmd, universal_newlines=True).rstrip().split('\n')
    if len(paths) == 1 and not paths[0]:
      app.Log(2, 'No files to import from %s', clone_dir)
      return
    app.Log(1, "Importing %s '%s' files from %s ...", humanize.Commas(
        len(paths)), importer.source_code_pattern, clone_dir.name)
    all_files_relpaths = public.GetAllFilesRelativePaths(clone_dir)
    jobs = [
        scrape_repos_pb2.ImportWorker(
            clone_from_url=repo.clone_from_url,
            clone_dir=str(clone_dir),
            abspath=p,
            all_files_relpaths=all_files_relpaths,
            preprocessors=importer.preprocessor,
        ) for p in paths
    ]
    bar = progressbar.ProgressBar(max_value=len(jobs))
    for outputs in bar(pool.imap_unordered(ImportWorker, jobs)):
      for output in outputs:
        session.add(output)


def ImportFromLanguage(db: contentfiles.ContentFiles,
                       language: scrape_repos_pb2.LanguageToClone,
                       pool: multiprocessing.Pool) -> None:
  """Import contentfiles from a language specification.

  Args:
    db: The database to import to.
    language: The language to import.
    pool: A multiprocessing pool.

  Raises:
    ValueError: If importer field not set.
  """
  if not language.importer:
    raise ValueError('LanguageToClone.importer field not set')

  with db.Session() as session:
    repos_to_import = [
        pathlib.Path(language.destination_directory / f)
        for f in pathlib.Path(language.destination_directory).iterdir()
        if ShouldImportRepo(session,
                            pathlib.Path(language.destination_directory / f))
    ]
  random.shuffle(repos_to_import)
  app.Log(1, 'Importing %s %s repos ...', humanize.Commas(len(repos_to_import)),
          language.language.capitalize())
  for metafile in repos_to_import:
    with db.Session(commit=True) as session:
      ImportRepo(session, language, metafile, pool)


def GetContentfilesDatabase(
    language: scrape_repos_pb2.LanguageToClone) -> contentfiles.ContentFiles:
  """Return a connection to the language database from clone list."""
  d = pathlib.Path(language.destination_directory)
  d = d.parent / (str(d.name) + '.db')
  return contentfiles.ContentFiles(f'sqlite:///{d}')


def GetImportMultiprocessingPool() -> multiprocessing.Pool:
  return multiprocessing.Pool(FLAGS.processes)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments '{}'".format(', '.join(argv[1:])))

  clone_list_path = pathlib.Path(FLAGS.importer_clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')
  clone_list = pbutil.FromFile(clone_list_path,
                               scrape_repos_pb2.LanguageCloneList())

  # Error early if the config contains invalid preprocessors.
  for language in clone_list.language:
    for importer in language.importer:
      [preprocessors.GetPreprocessorFunction(p) for p in importer.preprocessor]

  pool = GetImportMultiprocessingPool()
  for language in clone_list.language:
    db = GetContentfilesDatabase(language)
    if pathlib.Path(language.destination_directory).is_dir():
      ImportFromLanguage(db, language, pool)


if __name__ == '__main__':
  app.RunWithArgs(main)
