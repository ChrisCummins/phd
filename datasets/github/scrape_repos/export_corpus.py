"""Export ContentFiles to a directory."""
import binascii
import os
import pathlib

from sqlalchemy import orm

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import app
from labm8 import humanize
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string('clone_list', None, 'The path to a LanguageCloneList file.')
app.DEFINE_string('export_path', None, 'The root directory to export files to.')


def ExportDatabase(session: orm.session.Session,
                   export_path: pathlib.Path) -> None:
  """Export the contents of a database to a directory."""
  query = session.query(contentfiles.ContentFile)
  app.Log(1, 'Exporting %s files to %s ...', humanize.Commas(query.count()),
          export_path)
  for contentfile in query:
    path = export_path / (contentfile.sha256_hex + '.txt')
    app.Log(2, path)
    with open(path, 'w') as f:
      f.write(contentfile.text)


def ExportIndex(index_path: pathlib.Path, export_path: pathlib.Path) -> None:
  """Export the contents of an index directory to a directory."""
  contentfile = scrape_repos_pb2.ContentFile()
  for subdir, dirs, files in os.walk(index_path):
    for file in files:
      if file.endswith('.pbtxt'):
        try:
          pbutil.FromFile(pathlib.Path(os.path.join(subdir, file)), contentfile)
          sha256 = binascii.hexlify(contentfile.sha256).decode('utf-8')
          out_path = export_path / (sha256 + '.txt')
          if not out_path.is_file():
            with open(out_path, 'w') as f:
              f.write(contentfile.text)
              app.Log(2, out_path)
        except pbutil.DecodeError:
          pass


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

  # To export from contentfiles database.
  for language in clone_list.language:
    d = pathlib.Path(language.destination_directory)
    d = d.parent / (str(d.name) + '.db')
    db = contentfiles.ContentFiles(f'sqlite:///{d}')
    with db.Session() as session:
      (export_path / language.language).mkdir(exist_ok=True)
      ExportDatabase(session, export_path / language.language)

  # # To export from index directory.
  # for language in clone_list.language:
  #   index_path = pathlib.Path(language.destination_directory + '.index')
  #   if index_path.is_dir():
  #     (export_path / language.language).mkdir(exist_ok=True)
  #     ExportIndex(index_path, export_path / language.language)


if __name__ == '__main__':
  app.RunWithArgs(main)
