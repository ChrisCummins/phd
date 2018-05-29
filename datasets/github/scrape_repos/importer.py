"""Import files into a ContentFiles database."""
import pathlib
import subprocess

import humanize
from absl import app
from absl import flags
from absl import logging
from sqlalchemy import orm

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string('clone_list', None, 'The path to a LanguageCloneList file.')


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


def ImportRepo(session: orm.session.Session,
               language: scrape_repos_pb2.LanguageToClone,
               metafile: pathlib.Path) -> None:
  """Import contentfiles from repository.

  Args:
    session: A database session to import to.
    language: The language specification for the repo.
    metafile: The repo metafile.
  """
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  clone_dir = metafile.parent / f'{meta.owner}_{meta.name}'
  repo = contentfiles.GitHubRepository.GetOrAdd(session, meta)
  repo.language = language.language

  for importer in language.importer:
    if not importer.source_code_pattern:
      logging.error('No source_code_pattern specified! Stopping now.')
      return

    pat = importer.source_code_pattern
    pat = f'{clone_dir}/{pat[1:]}' if pat[0] == '^' else f'{clone_dir}/{pat}'
    cmd = ['find', str(clone_dir), '-type', 'f', '-regex', pat, '-not',
           '-path', '*/.git/*']
    paths = subprocess.check_output(
        cmd, universal_newlines=True).rstrip().split('\n')
    if len(paths) == 1 and not paths[0]:
      logging.debug('No files to import from %s', clone_dir)
      return
    logging.info('Importing %s %s files from %s ...',
                 humanize.intcomma(len(paths)),
                 language.language.capitalize(), clone_dir)
    for path in paths:
      try:
        if pathlib.Path(path).is_file():
          session.add(contentfiles.ContentFile.FromFile(meta, clone_dir, path))
      except UnicodeError:
        logging.warning('Failed to decode %s', path)


def ImportFromLanguage(db: contentfiles.ContentFiles,
                       language: scrape_repos_pb2.LanguageToClone) -> None:
  """Import contentfiles from a language specification.

  Args:
    db: The database to import to.
    language: The language to import.
  """
  with db.Session() as session:
    repos_to_import = [pathlib.Path(language.destination_directory / f) for f in
                       pathlib.Path(language.destination_directory).iterdir() if
                       ShouldImportRepo(session, pathlib.Path(
                           language.destination_directory / f))]
  logging.info('Importing %s %s repos ...',
               humanize.intcomma(len(repos_to_import)),
               language.language.capitalize())
  for metafile in repos_to_import:
    with db.Session(commit=True) as session:
      ImportRepo(session, language, metafile)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments '{}'".format(', '.join(argv[1:])))

  clone_list_path = pathlib.Path(FLAGS.clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')
  clone_list = pbutil.FromFile(clone_list_path,
                               scrape_repos_pb2.LanguageCloneList())

  for language in clone_list.language:
    d = pathlib.Path(language.destination_directory)
    d = d.parent / (str(d.name) + '.db')
    db = contentfiles.ContentFiles(d)
    if pathlib.Path(language.destination_directory).is_dir():
      ImportFromLanguage(db, language)


if __name__ == '__main__':
  app.run(main)
