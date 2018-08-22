"""Index ContentFiles from cloned GitHub repos."""
import multiprocessing
import os

import humanize
import pathlib
import random
from absl import app
from absl import flags
from absl import logging
from phd.lib.labm8 import pbutil

from datasets.github.scrape_repos import github_repo
from datasets.github.scrape_repos.preprocessors import preprocessors
from datasets.github.scrape_repos.proto import scrape_repos_pb2


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'indexer_processes', os.cpu_count(),
    'The number of indexer processes to run.')
flags.DEFINE_string(
    'clone_list', None,
    'The path to a LanguageCloneList file.')


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

  all_repos = [
    github_repo.GitHubRepo(pathlib.Path(language.destination_directory / f))
    for f in pathlib.Path(language.destination_directory).iterdir()
    if f.name.endswith('.pbtxt')]
  repos_to_import = [repo for repo in all_repos if not repo.IsIndexed()]
  random.shuffle(repos_to_import)
  logging.info('Importing %s %s repos ...',
               humanize.intcomma(len(repos_to_import)),
               language.language.capitalize())
  for repo in repos_to_import:
    repo.Index(list(language.importer), pool)


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
  app.run(main)
