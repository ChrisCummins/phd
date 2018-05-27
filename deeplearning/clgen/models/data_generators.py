"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""

import collections
import sys

import humanize
import numpy as np
from absl import flags
from absl import logging
from keras import utils

from deeplearning.clgen import corpuses
from deeplearning.clgen.proto import model_pb2


FLAGS = flags.FLAGS

# An <X,y> data tuple used for training one batch.
DataBatch = collections.namedtuple('DataBatch', ['X', 'y'])


class DataGeneratorBase(object):
  """The base class for training data generators."""

  def __init__(self, corpus: corpuses.Corpus,
               training_opts: model_pb2.TrainingOptions):
    self.corpus = corpus
    self.training_opts = training_opts
    self.shuffle = training_opts.shuffle_corpus_contentfiles_between_epochs
    self.encoded_corpus = self.corpus.GetTrainingData(shuffle=self.shuffle)
    logging.info('Encoded corpus: %s',
                 humanize.naturalsize(sys.getsizeof(self.encoded_corpus)))
    self.corpus_len = len(self.encoded_corpus)
    self.batch_size = min(training_opts.batch_size,
                          self.corpus_len - self.corpus.sequence_length - 1)
    if self.batch_size < training_opts.batch_size:
      logging.warning(
          'Requested training.batch_size (%d) is larger than the corpus (%d). '
          'Reduced batch size to %d', training_opts.batch_size, self.corpus_len,
          self.batch_size)

    # Set this publicly visibly attribute. The number of steps per epoch is
    # the total number of batches per epoch.
    self.steps_per_epoch = int(
        self.corpus_len / (self.batch_size * self.corpus.sequence_length))

  def __next__(self) -> DataBatch:
    raise NotImplementedError('DataGeneratorBase is abstract')

  def Vectorize(self, data: DataBatch) -> DataBatch:
    """One-hot encode a sequence of encoded tokens.

    Args:
      data: An X,y pair of encoded token sequences.

    Returns:
      One-hot encoded sequences.
    """
    # TODO(cec): Use keras to_categorical() instead of vectorizing by hand.
    _ = utils.to_categorical(data.y, self.corpus.vocabulary_size)

    X = np.zeros(
        (self.batch_size, self.corpus.sequence_length,
         self.corpus.vocabulary_size),
        dtype=np.bool)
    y = np.zeros((self.batch_size, self.corpus.vocabulary_size), dtype=np.bool)
    for i, sequence in enumerate(data.X):
      for t, encoded_char in enumerate(sequence):
        X[i, t, encoded_char] = 1
      y[i, data.y[i]] = 1
    return DataBatch(X=X, y=y)


class LazyVectorizingGenerator(DataGeneratorBase):
  """A data generator which vectorizes an encoded corpus in batches.

  This generator converts only enough of the encoded corpus into one-hot vectors
  as is necessary for a single batch. The memory requirement is
  O(batch_size * vocab_size). This is slower than one-hot encoding the entire
  corpus, but requires less memory.
  """

  def __init__(self, *args, **kwargs):
    super(LazyVectorizingGenerator, self).__init__(*args, **kwargs)
    self.skip = 1  # TODO(cec): Add this as a field in Model proto.

    # Start index into the encoded corpus.
    self.i = 0

    # Create a dummy batch of data to get the size of it.
    x = np.zeros(
        (self.batch_size, self.corpus.sequence_length,
         self.corpus.vocabulary_size),
        dtype=np.bool)
    y = np.zeros((self.batch_size, self.corpus.vocabulary_size), dtype=np.bool)
    batch_size = sys.getsizeof(x) + sys.getsizeof(y)
    logging.info('%s memory: %s per-batch, %s per-epoch, %s total',
                 type(self).__name__,
                 humanize.naturalsize(batch_size),
                 humanize.naturalsize(batch_size * self.steps_per_epoch),
                 humanize.naturalsize(batch_size * self.steps_per_epoch *
                                      self.training_opts.num_epochs))

  def __next__(self) -> DataBatch:
    """Generate the next batch of X, y pairs."""
    # Reset the position in the encoded corpus if we've run out of text.
    if (self.i + self.batch_size + self.corpus.sequence_length + 1 >=
        self.corpus_len):
      self.i = 0
      if self.shuffle:
        self.encoded_corpus = self.corpus.GetTrainingData(shuffle=True)

    # X_data = np.ndarray((self.batch_size, self.corpus.sequence_length),
    #                     dtype=np.int32)
    # y_data = np.ndarray((self.batch_size,), dtype=np.int32)
    X_data, y_data = [], []
    for i in range(self.i, self.i + self.batch_size, self.skip):
      sequence = np.array(
          self.encoded_corpus[i:i + self.corpus.sequence_length])
      next_token = self.encoded_corpus[i + self.corpus.sequence_length]
      # X_data[i] = sequence
      # y_data[i] = next_token
      X_data.append(sequence)
      y_data.append(next_token)

    logging.debug('%s produced %d sequences of length %d', type(self).__name__,
                  self.batch_size, self.corpus.sequence_length)

    self.i += self.batch_size
    return self.Vectorize(DataBatch(X=X_data, y=y_data))


def AutoGenerator(corpus: corpuses.Corpus,
                  training_opts: model_pb2.TrainingOptions) -> DataGeneratorBase:
  """Determine and construct what we believe to be the best data generator.

  The optimum generator will depend on the size of the corpus, the amount of
  memory available, and the vocabulary encoding.

  Args:
    corpus: A Corpus instance.
    training_opts: A TrainingOptions proto.

  Returns:
    A DataGenorator instance, ready to be used in a model's fit_generator()
    method.
  """
  return LazyVectorizingGenerator(corpus, training_opts)
