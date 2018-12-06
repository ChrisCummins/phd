"""Enumerate cached models."""
import pathlib

from absl import app
from absl import flags
from absl import logging
from phd.lib.labm8 import pbutil

from deeplearning.clgen.models import models
from deeplearning.clgen.proto import internal_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'working_dir', str(pathlib.Path('~/.cache/clgen').expanduser()),
    'The path to the CLgen working directory.')


def LsModels(cache_root: pathlib.Path) -> None:
  for model_dir in (cache_root / 'model').iterdir():
    meta_file = model_dir / 'META.pbtxt'
    if pbutil.ProtoIsReadable(meta_file, internal_pb2.ModelMeta()):
      model = models.Model(
          pbutil.FromFile(meta_file, internal_pb2.ModelMeta()).config)
      telemetry = list(model.TrainingTelemetry())
      num_epochs = model.config.training.num_epochs
      n = len(telemetry)
      print(f'{model_dir} {n} / {num_epochs} epochs')
    elif meta_file.is_file():
      logging.warning('Meta file %s cannot be read.', meta_file)
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
