"""This file is the entry point for creating chunks."""
import pathlib

from labm8 import app
from util.photolib.shutterbug import shutterbug

FLAGS = app.FLAGS

app.DEFINE_string(
    'src_dir', None,
    'The directory to create chunks from. All files in this directory are '
    'packed into chunks.')
app.DEFINE_string(
    'chunks_dir', None,
    'The root directory of the chunks. Each chunk is a directory containing '
    'files and a manifest.')
app.DEFINE_integer(
    'size_mb', 4695,
    'The smaximum size of each chunk in megabytes. This excludes the MANIFEST '
    'and README files which are generated.')
app.DEFINE_string('chunk_prefix', 'chunk_',
                  'The name to prepend to generated chunks.')
app.DEFINE_boolean(
    'random_ordering', True,
    'Whether to randomize the ordering of files across and within chunks. If '
    '--norandom_ordering is used, the files are arranged in chunks in the order '
    'in which they are found in --src_dir. This is not recommended, as it means '
    'the loss of a chunk causes a loss in a contiguous block of files.')
app.DEFINE_integer(
    'random_ordering_seed', 0,
    'The number used to seed the random number generator. Not used if '
    '--norandom_ordering is set. Using the same seed produces the same ordering '
    'of files.')
app.DEFINE_boolean(
    'gzip_files', False,
    'Whether to gzip individual files in chunks. Files are only stored in gzip '
    'form if it is smaller than the original file. For compressed image formats '
    'like JPGs, gzip rarely offers a reduction in file size.')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unknown flags "{}".'.format(', '.join(argv[1:])))

  src_dir = pathlib.Path(FLAGS.src_dir)
  if not src_dir.is_dir():
    raise app.UsageError('--src_dir not found')

  chunks_dir = pathlib.Path(FLAGS.chunks_dir)
  if not chunks_dir.is_dir():
    raise app.UsageError('--chunks_dir not found')

  size_in_bytes = FLAGS.size_mb * 1000**2
  if FLAGS.size_mb < 1:
    raise app.UsageError('--size_mb must be greater than or equal to one.')

  chunk_prefix = FLAGS.chunk_prefix
  if '/' in chunk_prefix:
    raise app.UsageError("--chunk_prefix cannot contain '/' character.")

  shutterbug.MakeChunks([src_dir],
                        chunks_dir,
                        size_in_bytes,
                        prefix=chunk_prefix,
                        shuffle=FLAGS.random_ordering,
                        seed=FLAGS.random_ordering_seed,
                        gzip=FLAGS.gzip_files)


if __name__ == '__main__':
  app.RunWithArgs(main)
