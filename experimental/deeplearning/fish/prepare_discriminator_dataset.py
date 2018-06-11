"""Create directories of training, test, and validation data."""
import collections
import pathlib

import humanize
from absl import app
from absl import flags
from absl import logging

from experimental.deeplearning.fish.proto import fish_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'export_path', '~/data/experimental/deeplearning/fish/75k',
    'Path to data exported by ./export_clang_opencl_dataset.')
flags.DEFINE_string(
    'dataset_root', '~/data/experimental/deeplearning/fish/crash_dataset',
    'Path to export training / validation / testing data to.')
flags.DEFINE_float(
    'training_ratio', 0.9, 'Ratio of dataset to use for training.')
flags.DEFINE_float(
    'validation_ratio', 0.0, 'Ratio of dataset to use for validation.')
flags.DEFINE_float(
    'testing_ratio', 0.1, 'Ratio of dataset to use for testing.')
flags.DEFINE_integer(
    'max_protos', 100000, 'The maximum number of protos per class to read')
flags.DEFINE_boolean(
    'assertions_only', False, 'If set, load only positive protos which raise '
    'compiler assertions.')
flags.DEFINE_boolean(
    'uneven_classes', False, 'If set, balance the number of positive and '
    'negative examples.')

DatasetRatios = collections.namedtuple(
    'DataSetRatios', ['train', 'val', 'test'])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if not FLAGS.export_path:
    raise app.UsageError('--export_path must be a directory')
  export_path = pathlib.Path(FLAGS.export_path)
  if export_path.is_file():
    raise app.UsageError('--export_path must be a directory')
  export_path.mkdir(parents=True, exist_ok=True)

  if not FLAGS.dataset_root:
    raise app.UsageError('--dataset_root must be a directory')
  dataset_root = pathlib.Path(FLAGS.dataset_root)
  if dataset_root.is_file():
    raise app.UsageError('--dataset_root must be a directory')
  dataset_root.mkdir(parents=True, exist_ok=True)

  ratios = DatasetRatios(
      FLAGS.training_ratio, FLAGS.validation_ratio, FLAGS.testing_ratio)
  assert sum(ratios) <= 1

  positive_protos = []
  i = 0
  for path in sorted(list((export_path / 'build_crash').iterdir())):
    proto = pbutil.FromFile(
        path, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
    if FLAGS.assertions_only and not proto.raised_assertion:
      continue
    i += 1
    positive_protos.append(proto)
    if i >= FLAGS.max_protos:
      break
  logging.info('Loaded %s positive data protos',
               humanize.intcomma(len(positive_protos)))
  # Load an equal number of negative protos, sorted by result ID.
  negative_proto_paths = sorted(list((export_path / 'pass').iterdir()))
  if not FLAGS.uneven_classes:
    negative_proto_paths = negative_proto_paths[:len(positive_protos)]
  negative_protos = [
    pbutil.FromFile(path, fish_pb2.CompilerCrashDiscriminatorTrainingExample())
    for path in negative_proto_paths
  ]
  logging.info('Loaded %s negative data protos',
               humanize.intcomma(len(negative_protos)))

  positive_sizes = [
      int(len(positive_protos) * FLAGS.training_ratio),
      int(len(positive_protos) * FLAGS.validation_ratio),
      int(len(positive_protos) * FLAGS.testing_ratio),
  ]
  negative_sizes = [
      int(len(negative_protos) * FLAGS.training_ratio),
      int(len(negative_protos) * FLAGS.validation_ratio),
      int(len(negative_protos) * FLAGS.testing_ratio),
  ]

  (dataset_root / 'training').mkdir(exist_ok=True, parents=True)
  (dataset_root / 'validation').mkdir(exist_ok=True, parents=True)
  (dataset_root / 'testing').mkdir(exist_ok=True, parents=True)

  for i, proto in enumerate(positive_protos[:positive_sizes[0]]):
    pbutil.ToFile(
        proto, (dataset_root / 'training' / f'positive-{i:04d}.pbtxt'))
  for i, proto in enumerate(negative_protos[:negative_sizes[0]]):
    pbutil.ToFile(
        proto, (dataset_root / 'training' / f'negative-{i:04d}.pbtxt'))
  logging.info('Wrote %s training examples',
               humanize.intcomma(positive_sizes[0] + negative_sizes[0]))
  positive_protos = positive_protos[positive_sizes[0]:]
  negative_protos = negative_protos[negative_sizes[0]:]

  for i, proto in enumerate(positive_protos[:positive_sizes[1]]):
    pbutil.ToFile(
        proto, (dataset_root / 'validation' / f'positive-{i:04d}.pbtxt'))
  for i, proto in enumerate(negative_protos[:negative_sizes[1]]):
    pbutil.ToFile(
        proto, (dataset_root / 'validation' / f'negative-{i:04d}.pbtxt'))
  logging.info('Wrote %s validation examples',
               humanize.intcomma(positive_sizes[1] + negative_sizes[1]))
  positive_protos = positive_protos[positive_sizes[1]:]
  negative_protos = negative_protos[negative_sizes[1]:]

  for i, proto in enumerate(positive_protos[:positive_sizes[2]]):
    pbutil.ToFile(
        proto, (dataset_root / 'testing' / f'positive-{i:04d}.pbtxt'))
  for i, proto in enumerate(negative_protos[:negative_sizes[2]]):
    pbutil.ToFile(
        proto, (dataset_root / 'testing' / f'negative-{i:04d}.pbtxt'))
  logging.info('Wrote %s testing examples',
               humanize.intcomma(positive_sizes[2] + negative_sizes[2]))


if __name__ == '__main__':
  app.run(main)
