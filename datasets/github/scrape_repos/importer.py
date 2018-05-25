"""Import files into a ContentFiles database."""
import pathlib
import subprocess

from absl import app
from absl import flags
from absl import logging
from sqlalchemy import orm

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string('clone_list', None, 'The path to a LanguageCloneList file.')


def ShouldImport(session: orm.session.Session, metafile: pathlib.Path) -> bool:
  if not (metafile.is_file() and pbutil.ProtoIsReadable(metafile,
                                                        scrape_repos_pb2.GitHubRepoMetadata())):
    return False
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  clone_dir = metafile.parent / f'{meta.owner}_{meta.name}'
  if not (clone_dir / '.git').is_dir():
    return False
  return not contentfiles.GitHubRepository.IsInDatabase(session, meta)


FILE_EXTENSIONS = {
  'glsl': ['.glsl', '.frag', '.vert', '.tesc', '.tese', '.geom', '.comp'],
  'opencl': ['.cl', '.ocl'], 'solidity': ['.sol'], 'javascript': ['.js'],
  'c': ['.c', '.h', '.inc'], 'go': ['.go'], 'java': ['.java'],
  'python': ['.py'], }


def ImportFromMetafile(db: contentfiles.ContentFiles,
                       language: scrape_repos_pb2.LanguageToClone,
                       metafile: pathlib.Path):
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  clone_dir = metafile.parent / f'{meta.owner}_{meta.name}'
  with db.Session(commit=True) as s:
    repo = contentfiles.GitHubRepository.GetOrAdd(s, meta)
    repo.language = language.language
    name_str = " -o ".join(
      [f"-name '*{ext}'" for ext in FILE_EXTENSIONS[language.language]])
    paths = subprocess.check_output(
      f"find {clone_dir} -type f {name_str} | grep -v '.git/' || true",
      shell=True, universal_newlines=True).rstrip().split('\n')
    if len(paths) == 1 and not paths[0]:
      logging.debug('No files to import from %s', clone_dir)
      return
    logging.info('Importing %s files from %s ...', len(paths), clone_dir)
    for path in paths:
      try:
        s.add(contentfiles.ContentFile.FromFile(meta, clone_dir, path))
      except UnicodeError:
        logging.warning('Failed to decode %s', path)


def ImportFromLanguage(db: contentfiles.ContentFiles,
                       language: scrape_repos_pb2.LanguageToClone) -> None:
  if not language.language in FILE_EXTENSIONS:
    logging.error('Language %s not supported! Importing nothing',
                  language.language)
  with db.Session() as s:
    repos_to_import = [pathlib.Path(language.destination_directory / f) for f in
                       pathlib.Path(language.destination_directory).iterdir() if
                       ShouldImport(s, pathlib.Path(
                         language.destination_directory / f))]
  logging.info('Importing %d %s repos ...', len(repos_to_import),
               language.language)
  for metafile in repos_to_import:
    ImportFromMetafile(db, language, metafile)


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
    d = d.parents / d.name + '.db'
    db = contentfiles.ContentFiles(d)
    if pathlib.Path(language.destination_directory).is_dir():
      ImportFromLanguage(db, language)


if __name__ == '__main__':
  app.run(main)
