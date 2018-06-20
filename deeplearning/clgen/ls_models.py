"""Enumerate cached models."""
import pathlib

from absl import app
from absl import flags
from absl import logging

from deeplearning.clgen.models import models
from deeplearning.clgen.proto import model_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'working_dir', str(pathlib.Path('~/.cache/clgen').expanduser()),
    'The path to the CLgen working directory.')


def LsModels(cache_root: pathlib.Path) -> None:
  for model_dir in (cache_root / 'model').iterdir():
    meta_file = model_dir / 'META.pbtxt'
    if pbutil.ProtoIsReadable(meta_file, model_pb2.Model()):
      model = models.Model(pbutil.FromFile(meta_file, model_pb2.Model()))
      telemetry = model.TrainingTelemetry()
      num_epochs = model.config.training.num_epochs
      print('%s %d / %d epochs', model_dir, len(telemetry), num_epochs)
    else:
      logging.warning('Meta file %s not found.', meta_file)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.working_dir:
    raise app.UsageError('--working_dir must be set')
  working_dir = pathlib.Path(FLAGS.working_dir)
  if working_dir.exists() and not working_dir.is_dir():
    raise app.UsageError('--working_dir must be a directory')

  LsModels(working_dir)


if __name__ == '__main__':
  app.run(main)
