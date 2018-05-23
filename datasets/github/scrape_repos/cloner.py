"""Clone GitHub repositories.

This looks for repo meta files and clones any which have not been cloned.
"""
import pathlib
import subprocess

import progressbar
from absl import app
from absl import flags
from absl import logging

from datasets.github.scrape_repos.proto import scrape_repos_pb2
from lib.labm8 import fs
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string('clone_list', None, 'The path to a LanguageCloneList file.')
flags.DEFINE_integer('repository_clone_timeout_minutes', 30,
                     'The maximum number of minutes to attempt to clone a '
                     'repository before '
                     'quitting and moving on to the next repository.')


def CloneFromMetafile(metafile: pathlib.Path) -> None:
  meta = pbutil.FromFile(metafile, scrape_repos_pb2.GitHubRepoMetadata())
  clone_dir = metafile.parent / f'{meta.owner}_{meta.name}'
  logging.debug('%s', meta)
  if not (clone_dir / '.git').is_dir():
    cmd = ['timeout', f'{FLAGS.repository_clone_timeout_minutes}m',
           '/usr/bin/git', 'clone', '--recursive', meta.clone_from_url,
           str(clone_dir)]
    logging.debug('$ %s', ' '.join(cmd))
    try:
      subprocess.check_call(cmd, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
      logging.warning('\nClone failed %s', clone_dir)
      fs.rm(clone_dir)


def IsRepoMetaFile(f: str):
  """Determine if a path is a GitHubRepoMetadata message."""
  return (fs.isfile(f) and pbutil.ProtoIsReadable(f,
                                                  scrape_repos_pb2.GitHubRepoMetadata()))


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  clone_list_path = pathlib.Path(FLAGS.clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')

  clone_list = pbutil.FromFile(clone_list_path,
                               scrape_repos_pb2.LanguageCloneList())

  meta_files = []
  for language in clone_list.language:
    directory = pathlib.Path(language.destination_directory)
    if directory.is_dir():
      meta_files += [pathlib.Path(directory / f) for f in directory.iterdir() if
                     IsRepoMetaFile(f)]
  for meta_file in progressbar.ProgressBar()(meta_files):
    CloneFromMetafile(meta_file)


if __name__ == '__main__':
  app.run(main)
