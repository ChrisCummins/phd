"""Import from //experimental/deeplearning/fish training protos."""
import pathlib
import tempfile
import typing

from absl import app
from absl import flags
from absl import logging

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from experimental.deeplearning.fish.proto import fish_pb2
from labm8 import pbutil


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'protos_dir', None,
    'Path to directory containing kernels to import.')
flags.DEFINE_string(
    'origin', None,
    'Name of the origin of the kernels, e.g. "github".')
flags.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to import OpenCL kernels to.')
flags.DEFINE_integer(
    'batch_size', 512, 'Number of protos to process at once.')


def CreateTempFileFromProto(
    tempdir: pathlib.Path,
    proto_path: pathlib.Path) -> pathlib.Path:
  """Write testcase to a file in directory."""
  proto = pbutil.FromFile(
      proto_path, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
  path = tempdir / proto_path.name
  with open(path, 'w') as f:
    f.write(proto.src)
  return path


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)
  protos_dir = pathlib.Path(FLAGS.protos_dir)
  if not protos_dir.is_dir():
    raise app.UsageError('Proto dir not found')

  if not FLAGS.origin:
    raise app.UsageError('Must set an origin')

  paths_to_import = list(protos_dir.iterdir())
  if not all(x.is_file() for x in paths_to_import):
    raise app.UsageError('Non-file input found')

  for stride in range(0, len(paths_to_import), FLAGS.batch_size):
    with tempfile.TemporaryDirectory(prefix='phd_fish_') as d:
      d = pathlib.Path(d)
      srcs = [CreateTempFileFromProto(d, p) for p in
              paths_to_import[stride:stride + FLAGS.batch_size]]
      db.ImportStaticFeaturesFromPaths(srcs, FLAGS.origin)
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
