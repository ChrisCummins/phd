"""Export ContentFiles to a directory."""
import humanize
import pathlib
from absl import app
from absl import flags
from absl import logging
from phd.lib.labm8 import pbutil
from sqlalchemy import orm

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.proto import scrape_repos_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string('clone_list', None,
                    'The path to a LanguageCloneList file.')
flags.DEFINE_string('export_path', None,
                    'The root directory to export files to.')


def ExportDatabase(session: orm.session.Session,
                   export_path: pathlib.Path) -> None:
  """Export the contents of a database to a directory."""
  query = session.query(contentfiles.ContentFile)
  logging.info('Exporting %s files to %s ...', humanize.intcomma(query.count()),
               export_path)
  for contentfile in query:
    path = export_path / (contentfile.sha256_hex + '.txt')
    logging.debug(path)
    with open(path, 'w') as f:
      f.write(contentfile.text)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments '{}'".format(', '.join(argv[1:])))

  clone_list_path = pathlib.Path(FLAGS.clone_list or '')
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')
  clone_list = pbutil.FromFile(clone_list_path,
                               scrape_repos_pb2.LanguageCloneList())

  if not FLAGS.export_path:
    raise app.UsageError('--export_path not set.')
  export_path = pathlib.Path(FLAGS.export_path)
  export_path.mkdir(parents=True, exist_ok=True)

  for language in clone_list.language:
    d = pathlib.Path(language.destination_directory)
    d = d.parent / (str(d.name) + '.db')
    db = contentfiles.ContentFiles(d)
    with db.Session() as session:
      (export_path / language.language).mkdir(exist_ok=True)
      ExportDatabase(session, export_path / language.language)


if __name__ == '__main__':
  app.run(main)
