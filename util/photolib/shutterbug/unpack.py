"""This file is the entry point for unpacking chunks."""
import pathlib

from labm8 import app
from util.photolib.shutterbug import shutterbug

FLAGS = app.FLAGS

app.DEFINE_string(
    'chunks_dir', None,
    'The root directory of the chunks. Each chunk is a directory containing '
    'files and a manifest.')
app.DEFINE_string(
    'out_dir', None,
    'The directory to write the unpacked chunks to. Each chunk contains files '
    'which are unpacked to a path relative to this directory.')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unknown flags "{}".'.format(', '.join(argv[1:])))

  chunks_dir = pathlib.Path(FLAGS.chunks_dir)
  if not chunks_dir.is_dir():
    raise app.UsageError('--chunks_dir not found')

  out_dir = pathlib.Path(FLAGS.out_dir)
  if not out_dir.is_dir():
    raise app.UsageError('--out_dir not found')

  shutterbug.unchunk(chunks_dir, out_dir)


if __name__ == '__main__':
  app.RunWithArgs(main)
