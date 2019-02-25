"""Create directories of training, test, and validation data."""
import collections
import pathlib
import random
import typing

import humanize
import numpy as np
from absl import app
from absl import flags
from absl import logging

from experimental.deeplearning.fish.proto import fish_pb2
from labm8 import labtypes
from labm8 import pbutil

FLAGS = flags.FLAGS

flags.DEFINE_string('export_path', '~/data/experimental/deeplearning/fish/75k',
                    'Path to data exported by ./export_clang_opencl_dataset.')
flags.DEFINE_string('dataset_root',
                    '~/data/experimental/deeplearning/fish/crash_dataset',
                    'Path to export training / validation / testing data to.')
flags.DEFINE_list('positive_class_outcomes', ['build_crash'],
                  'The outcomes to select positive examples from.')
flags.DEFINE_list('negative_class_outcomes', ['pass'],
                  'The outcomes to select negative examples from.')
flags.DEFINE_float('training_ratio', 0.9,
                   'Ratio of dataset to use for training.')
flags.DEFINE_float('validation_ratio', 0.0,
                   'Ratio of dataset to use for validation.')
flags.DEFINE_float('testing_ratio', 0.1, 'Ratio of dataset to use for testing.')
flags.DEFINE_integer('max_protos', 100000,
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
flags.DEFINE_integer(
    'max_src_len', 10000,
    'Ignore programs whose sources are longer than this number of characters')
flags.DEFINE_integer('seed', 0, 'Random seed to use when splitting data.')

# The size ratios to use for training, validation, and testing. Must sum to 1.
DatasetRatios = collections.namedtuple('DataSetRatios',
                                       ['train', 'val', 'test'])

DatasetSizes = collections.namedtuple('DataSetSizes', ['train', 'val', 'test'])

# Type alias for training protos.
TrainingProto = fish_pb2.CompilerCrashDiscriminatorTrainingExample


def GetProtos(export_path: pathlib.Path, outcomes: typing.List[str],
              max_src_len: int) -> typing.List[TrainingProto]:
  paths = sorted(
      labtypes.flatten(
          [list((export_path / outcome).iterdir()) for outcome in outcomes]))
  protos = []
  for path in paths:
    proto = pbutil.FromFile(path, TrainingProto())
    if len(proto.src) > max_src_len:
      continue
    protos.append(proto)
  return protos


def LoadPositiveProtos(export_path: pathlib.Path,
                       positive_class_outcomes: typing.List[str],
                       max_src_len: int, max_num: int,
                       assertions_only: bool) -> typing.List[TrainingProto]:
  """Load positive training protos."""
  protos = [
      p for p in GetProtos(export_path, positive_class_outcomes, max_src_len)
      if (not assertions_only) or p.raised_assertion
  ][:max_num]
  logging.info('Loaded %s positive data protos.', humanize.intcomma(
      len(protos)))
  return protos


def LoadNegativeProtos(
    export_path: pathlib.Path, positive_protos: typing.List[TrainingProto],
    negative_class_outcomes: typing.List[str], max_src_len: int,
    balance_class_lengths: bool,
    balance_class_counts: bool) -> typing.List[TrainingProto]:
  """Load negative training protos."""
  candidate_protos = GetProtos(export_path, negative_class_outcomes,
                               max_src_len)

  if balance_class_lengths:
    positive_proto_sizes = [len(p.src) for p in positive_protos]
    logging.info('Loaded %s negative protos. Balancing lengths ...',
                 humanize.intcomma(len(candidate_protos)))
    negative_proto_sizes = np.array([len(p.src) for p in candidate_protos],
                                    dtype=np.int32)
    negative_protos = []
    for i, positive_proto_size in enumerate(positive_proto_sizes):
      size_diffs = np.abs(negative_proto_sizes - positive_proto_size)
      idx_of_closest: int = np.argmin(size_diffs)
      logging.info(
          'Found negative example of size %s to match positive '
          'example of size %s (diff %s)',
          humanize.intcomma(negative_proto_sizes[idx_of_closest]),
          humanize.intcomma(positive_proto_size), size_diffs.min())
      negative_protos.append(candidate_protos[idx_of_closest])
      negative_proto_sizes = np.delete(negative_proto_sizes, [idx_of_closest])
      del candidate_protos[idx_of_closest]
      if not candidate_protos:
        logging.warning('Ran out of negative examples to choose from!')
        break
    positive_protos = positive_protos[:i]
  else:
    if balance_class_counts:
      min_count = min(len(positive_protos), len(negative_protos))
      candidate_protos = candidate_protos[:min_count]
      positive_protos = positive_protos[:min_count]
    negative_protos = candidate_protos
  logging.info('Loaded %s negative data protos',
               humanize.intcomma(len(negative_protos)))
  return positive_protos, negative_protos


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

  ratios = DatasetRatios(FLAGS.training_ratio, FLAGS.validation_ratio,
                         FLAGS.testing_ratio)
  assert sum(ratios) <= 1

  # Load protos.
  positive_protos = LoadPositiveProtos(
      export_path, FLAGS.positive_class_outcomes, FLAGS.max_src_len,
      FLAGS.max_protos, FLAGS.assertions_only)
  positive_protos, negative_protos = LoadNegativeProtos(
      export_path, positive_protos, FLAGS.negative_class_outcomes,
      FLAGS.max_src_len, FLAGS.balance_class_lengths,
      FLAGS.balance_class_counts)

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

  logging.info('Shuffling protos with seed %d', FLAGS.seed)
  random.seed(FLAGS.seed)
  random.shuffle(positive_protos)
  random.shuffle(negative_protos)

  for i, proto in enumerate(positive_protos[:positive_sizes[0]]):
    pbutil.ToFile(proto,
                  (dataset_root / 'training' / f'positive-{i:04d}.pbtxt'))
  for i, proto in enumerate(negative_protos[:negative_sizes[0]]):
    pbutil.ToFile(proto,
                  (dataset_root / 'training' / f'negative-{i:04d}.pbtxt'))
  logging.info('Wrote %s training examples',
               humanize.intcomma(positive_sizes[0] + negative_sizes[0]))
  positive_protos = positive_protos[positive_sizes[0]:]
  negative_protos = negative_protos[negative_sizes[0]:]

  for i, proto in enumerate(positive_protos[:positive_sizes[1]]):
    pbutil.ToFile(proto,
                  (dataset_root / 'validation' / f'positive-{i:04d}.pbtxt'))
  for i, proto in enumerate(negative_protos[:negative_sizes[1]]):
    pbutil.ToFile(proto,
                  (dataset_root / 'validation' / f'negative-{i:04d}.pbtxt'))
  logging.info('Wrote %s validation examples',
               humanize.intcomma(positive_sizes[1] + negative_sizes[1]))
  positive_protos = positive_protos[positive_sizes[1]:]
  negative_protos = negative_protos[negative_sizes[1]:]

  for i, proto in enumerate(positive_protos[:positive_sizes[2]]):
    pbutil.ToFile(proto, (dataset_root / 'testing' / f'positive-{i:04d}.pbtxt'))
  for i, proto in enumerate(negative_protos[:negative_sizes[2]]):
    pbutil.ToFile(proto, (dataset_root / 'testing' / f'negative-{i:04d}.pbtxt'))
  logging.info('Wrote %s testing examples',
               humanize.intcomma(positive_sizes[2] + negative_sizes[2]))


if __name__ == '__main__':
  app.run(main)
