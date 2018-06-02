"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""

import collections
import sys
import time
import typing

import humanize
import numpy as np
from absl import flags
from absl import logging

from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.proto import model_pb2


FLAGS = flags.FLAGS

# An <X,y> data tuple used for training one batch.
DataBatch = collections.namedtuple('DataBatch', ['X', 'y'])


def AutoGenerator(
    corpus: corpuses.Corpus,
    training_opts: model_pb2.TrainingOptions) -> typing.Generator[
  DataBatch, typing.Any, None]:
  """Determine and construct what we believe to be the best data generator.

  The optimum generator will depend on the corpus, the amount of memory
  available, and the vocabulary encoding.

  Args:
    corpus: A Corpus instance.
    training_opts: A TrainingOptions proto.

  Returns:
    A generator suitable for use by a model's fit_generator() method.
  """
  return BatchGenerator(corpus, training_opts)


def BatchGenerator(
    corpus: corpuses.Corpus,
    training_opts: model_pb2.TrainingOptions) -> typing.Generator[
  DataBatch, typing.Any, None]:
  """A batch generator which lazily one-hot encodes the y vectors.

  This reduces the memory overhead by only one-hot encoding the y vectors on a
  per-batch basis. This is of course slower than one-hot encoding the entire
  y corpus, but that requires more memory than is available on many systems for
  a reasonable corpus.

  Args:
    corpus: A Corpus instance.
    training_opts: A TrainingOptions proto.

  Returns:
    A generator suitable for use by a model's fit_generator() method.
  """
  x, y, steps_per_epoch = GetTrainingCorpus(corpus, training_opts)

  # Per-epoch outer loop.
  epoch_num = 0
  while True:
    # Re-shuffle corpus if needed.
    if epoch_num and training_opts.shuffle_corpus_contentfiles_between_epochs:
      x, y, steps_per_epoch = GetTrainingCorpus(corpus, training_opts)

    # Roll so that we don't need to reset model states over epochs.
    x_epoch = np.split(np.roll(x, -epoch_num, axis=0), steps_per_epoch, axis=1)
    y_epoch = np.split(np.roll(y, -epoch_num, axis=0), steps_per_epoch, axis=1)
    # Per-batch inner loop.
    for batch_num in range(steps_per_epoch):
      batch = DataBatch(
          X=x_epoch[batch_num],
          # Lazy one-hot encoding.
          y=OneHotEncode(y_epoch[batch_num], corpus.vocab_size))
      if not batch_num and not epoch_num:
        LogBatchTelemetry(batch, steps_per_epoch, training_opts.num_epochs)
      yield batch
    epoch_num += 1


def GetTrainingCorpus(
    corpus: corpuses.Corpus,
    training_opts: model_pb2.TrainingOptions) -> typing.Tuple[
  np.ndarray, np.ndarray, int]:
  """Get the corpus to train over.

  Args:
    corpus: A Corpus instance.
    training_opts: A TrainingOptions proto.

  Returns:
    An X, y pair of data for an epoch, and the number of steps in the epoch.

  Raises:
    UserError: If batch_size and sequence_length are too large for the corpus,
      yielding no batches.
  """
  start_time = time.time()
  encoded_corpus = corpus.GetTrainingData(
      shuffle=training_opts.shuffle_corpus_contentfiles_between_epochs)
  corpus_length = len(encoded_corpus)
  steps_per_epoch = (corpus_length - 1) // (
      training_opts.batch_size * training_opts.sequence_length)
  if not steps_per_epoch:
    raise errors.UserError(
        f'Requested batch size ({training_opts.batch_size}) and '
        f'sequence length ({training_opts.sequence_length}) are too large for '
        f'corpus of size {corpus_length}.')

  clipped_corpus_length = (
      steps_per_epoch * training_opts.batch_size *
      training_opts.sequence_length)

  x = np.reshape(
      encoded_corpus[:clipped_corpus_length],
      [training_opts.batch_size,
       steps_per_epoch * training_opts.sequence_length])
  y = np.reshape(
      encoded_corpus[1:clipped_corpus_length + 1],
      [training_opts.batch_size,
       steps_per_epoch * training_opts.sequence_length])

  logging.info(
      'Encoded corpus of %s tokens (clipped last %s tokens) in %s ms.',
      humanize.intcomma(clipped_corpus_length),
      humanize.intcomma(corpus_length - clipped_corpus_length),
      humanize.intcomma(int((time.time() - start_time) * 1000)))
  return x, y, steps_per_epoch


def OneHotEncode(indices: np.ndarray, vocabulary_size: int):
  """One-hot encode an array of vocabulary indices.

    Args:
      indices: A 1D array of vocabulary indices.
      vocabulary_size: The size of the vocabulary.

    Returns:
      A 2D array of one-hot encoded tokens.
    """
  return np.eye(vocabulary_size)[indices]


def LogBatchTelemetry(batch: DataBatch, steps_per_epoch: int,
                      num_epochs: int) -> None:
  """Log analytics about the batch."""
  logging.info("Step shape: X: %s, y"
               ": %s.", batch.X.shape, batch.y.shape)
  # sys.getsizeof() includes only the memory required for an object, not any
  # objects it refernces, so we must manually sum the X and y arrays.
  batch_size = sys.getsizeof(batch) + batch.X.nbytes + batch.y.nbytes
  logging.info(
      'Memory: %s per batch, %s per epoch, %s total.',
      humanize.naturalsize(batch_size),
      humanize.naturalsize(batch_size * steps_per_epoch),
      humanize.naturalsize(batch_size * steps_per_epoch * num_epochs))
