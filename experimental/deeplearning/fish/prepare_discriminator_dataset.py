"""Create directories of training, test, and validation data."""
import collections
import pathlib
import typing

import humanize
import numpy as np
from absl import app
from absl import flags
from absl import logging

from experimental.deeplearning.fish.proto import fish_pb2
from lib.labm8 import labtypes
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'export_path', '~/data/experimental/deeplearning/fish/75k',
    'Path to data exported by ./export_clang_opencl_dataset.')
flags.DEFINE_string(
    'dataset_root', '~/data/experimental/deeplearning/fish/crash_dataset',
    'Path to export training / validation / testing data to.')
flags.DEFINE_float(
    'training_ratio', 0.9,
    'Ratio of dataset to use for training.')
flags.DEFINE_float(
    'validation_ratio', 0.0,
    'Ratio of dataset to use for validation.')
flags.DEFINE_float(
    'testing_ratio', 0.1,
    'Ratio of dataset to use for testing.')
flags.DEFINE_integer(
    'max_protos', 100000,
    'The maximum number of protos per class to read')
flags.DEFINE_boolean(
    'assertions_only', False,
    'If set, load only positive protos which raise compiler assertions.')
flags.DEFINE_boolean(
    'balance_class_counts', False,
    'If set, balance the number of positive and negative examples. Has no '
    'effect if --balance_class_lengths flag is used.')
flags.DEFINE_boolean(
    'balance_class_lengths', False,
    'If set, balance the length of positive and negative examples. For every '
    'positive example in the dataset, the negative example with the closest '
    'length is added.')
flags.DEFINE_boolean(
    'include_bf_outcomes_as_negative', False,
    'If set, a build failure outcome is considered a negative training '
    'example. If not set, only pass outcomes are counted as negative examples.')

# The size ratios to use for training, validation, and testing. Must sum to 1.
DatasetRatios = collections.namedtuple(
    'DataSetRatios', ['train', 'val', 'test'])

DatasetSizes = collections.namedtuple(
    'DataSetSizes', ['train', 'val', 'test'])

# Type alias for training protos.
TrainingProto = fish_pb2.CompilerCrashDiscriminatorTrainingExample


def LoadPositiveProtos(
    export_path: pathlib.Path, max_num: int) -> typing.List[TrainingProto]:
  """Load positive training protos."""
  outputs = []
  for path in sorted(list((export_path / 'build_crash').iterdir()))[:max_num]:
    proto = pbutil.FromFile(
        path, TrainingProto())
    if FLAGS.assertions_only and not proto.raised_assertion:
      continue
    outputs.append(proto)
  logging.info('Loaded %s positive data protos.',
               humanize.intcomma(len(outputs)))
  return outputs


def GetNegativeExampleDirs(
    export_path: pathlib.Path,
    include_bf_outcomes_as_negative: bool) -> typing.List[pathlib.Path]:
  """Get the list of directories to load negative training examples from."""
  dirs = [(export_path / 'pass')]
  if include_bf_outcomes_as_negative:
    dirs.append((export_path / 'build_failure'))
  return dirs


def LoadNegativeProtos(
    export_path: pathlib.Path, positive_protos: typing.List[TrainingProto],
    balance_class_lengths: bool,
    balance_class_counts: bool,
    include_bf_outcomes_as_negative: bool) -> typing.List[TrainingProto]:
  """Load negative training protos."""
  negative_proto_paths = sorted(
      labtypes.flatten([
        list(d.iterdir()) for d in
        GetNegativeExampleDirs(export_path, include_bf_outcomes_as_negative)]))

  if balance_class_lengths:
    positive_proto_sizes = [len(p.src) for p in positive_protos]
    negative_protos = [
      pbutil.FromFile(path, TrainingProto())
      for path in list((export_path / 'pass').iterdir())
    ]
    logging.info('Loaded %s negative protos. Balancing lengths ...',
                 humanize.intcomma(len(negative_protos)))
    negative_proto_sizes = np.array(
        [len(p.src) for p in negative_protos], dtype=np.int32)
    outputs = []
    for positive_proto_size in positive_proto_sizes:
      size_diffs = np.abs(negative_proto_sizes - positive_proto_size)
      idx_of_closest: int = np.argmin(size_diffs)
      logging.info('Found negative example of size %s to match positive '
                   'example of size %s (diff %s)',
                   humanize.intcomma(negative_proto_sizes[idx_of_closest]),
                   humanize.intcomma(positive_proto_size),
                   size_diffs.min())
      outputs.append(negative_protos[idx_of_closest])
      negative_proto_sizes = np.delete(negative_proto_sizes, [idx_of_closest])
      del negative_protos[idx_of_closest]
      if not negative_protos:
        logging.warning('Ran out of negative examples to choose from!')
        break
  else:
    if balance_class_counts:
      # Limit the number of negative examples to <= number of positive.
      negative_proto_paths = negative_proto_paths[:len(positive_protos)]
      if len(negative_proto_paths) < len(positive_protos):
        logging.warning('Fewer negative examples (%s) than positive (%s)!',
                        humanize.intcomma(len(negative_proto_paths)),
                        humanize.intcomma(len(positive_protos)))
    outputs = [
      pbutil.FromFile(path, TrainingProto())
      for path in negative_proto_paths
    ]
  logging.info('Loaded %s negative data protos',
               humanize.intcomma(len(outputs)))
  return outputs


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

  # Load protos.
  positive_protos = LoadPositiveProtos(export_path, FLAGS.max_protos)
  negative_protos = LoadNegativeProtos(
      export_path, positive_protos, FLAGS.balance_class_lengths,
      FLAGS.balance_class_counts, FLAGS.include_bf_outcomes_as_negative)

  positive_sizes = DatasetSizes(
      int(len(positive_protos) * FLAGS.training_ratio),
      int(len(positive_protos) * FLAGS.validation_ratio),
      int(len(positive_protos) * FLAGS.testing_ratio),
  )
  negative_sizes = DatasetSizes(
      int(len(negative_protos) * FLAGS.training_ratio),
      int(len(negative_protos) * FLAGS.validation_ratio),
      int(len(negative_protos) * FLAGS.testing_ratio),
  )

  # Create output directories.
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
