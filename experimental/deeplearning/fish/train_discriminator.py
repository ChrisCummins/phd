"""Train a discriminator."""
import pathlib

import humanize
from absl import app
from absl import flags
from absl import logging
from experimental.fish.proto import fish_pb2

from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'training_path', None,
    'Directory to read training data from.')


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.export_path:
    raise app.UsageError('--export_path must be a directory')
  training_data_path = pathlib.Path(FLAGS.export_path)
  if training_data_path.is_file():
    raise app.UsageError('--export_path must be a directory')
  training_data_path.mkdir(parents=True, exist_ok=True)

  training_protos = [
    pbutil.FromFile(path, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
    for path in training_data_path.iterdir()
  ]

  logging.info('Loaded %s training data protos',
               humanize.intcomma(len(training_protos)))


if __name__ == '__main__':
  app.run(main)
