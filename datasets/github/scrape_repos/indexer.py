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
"""Index ContentFiles from cloned GitHub repos."""
import multiprocessing
import os
import pathlib
import random

from datasets.github.scrape_repos import github_repo
from datasets.github.scrape_repos.preprocessors import preprocessors
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import app
from labm8 import humanize
from labm8 import pbutil

FLAGS = app.FLAGS
app.DEFINE_integer('indexer_processes', os.cpu_count(),
                   'The number of indexer processes to run.')
app.DEFINE_string('clone_list', None, 'The path to a LanguageCloneList file.')


def ImportFromLanguage(language: scrape_repos_pb2.LanguageToClone,
                       pool: multiprocessing.Pool) -> None:
  """Import contentfiles from a language specification.

  Args:
    language: The language to import.
    pool: A multiprocessing pool.

  Raises:
    ValueError: If importer field not set.
  """
  if not language.importer:
    raise ValueError('LanguageToClone.importer field not set')

  app.Log(1, 'Enumerating all repos ...')
  all_repos = [
      github_repo.GitHubRepo(pathlib.Path(language.destination_directory / f))
      for f in pathlib.Path(language.destination_directory).iterdir()
      if f.name.endswith('.pbtxt')
  ]
  app.Log(1, 'Pruning indexed repos ...')
  num_repos = len(all_repos)
  repos_to_import = [repo for repo in all_repos if not repo.IsIndexed()]
  num_todo = len(repos_to_import)
  num_pruned = num_repos - num_todo
  random.shuffle(repos_to_import)
  app.Log(1, 'Importing %s of %s %s repos ...', humanize.Commas(num_todo),
          humanize.Commas(num_repos), language.language.capitalize())
  for i, repo in enumerate(repos_to_import):
    repo.Index(
        list(language.importer), pool,
        github_repo.IndexProgress(num_pruned + i, num_repos))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments '{}'".format(', '.join(argv[1:])))

  clone_list_path = pathlib.Path(FLAGS.clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')
  clone_list = pbutil.FromFile(clone_list_path,
                               scrape_repos_pb2.LanguageCloneList())

  # Error early if the config contains invalid preprocessors.
  for language in clone_list.language:
    for importer in language.importer:
      [preprocessors.GetPreprocessorFunction(p) for p in importer.preprocessor]

  pool = multiprocessing.Pool(FLAGS.indexer_processes)
  for language in clone_list.language:
    ImportFromLanguage(language, pool)


if __name__ == '__main__':
  app.RunWithArgs(main)
