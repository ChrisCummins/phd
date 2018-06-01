"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""

import collections
import sys
import time

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


class DataGeneratorBase(object):
  """The base class for training data generators."""

  def __init__(self, corpus: corpuses.Corpus,
               training_opts: model_pb2.TrainingOptions):
    """Instantiate a data generator.

    Args:
      corpus: A Corpus instance.
      training_opts: A TrainingOptions config proto.

    Raises:
      UserError: If the corpus is smaller than the sequence length.
    """
    self.corpus = corpus
    self.training_opts = training_opts
    self.shuffle = training_opts.shuffle_corpus_contentfiles_between_epochs
    start_time = time.time()
    self.encoded_corpus = self.corpus.GetTrainingData(shuffle=self.shuffle)
    logging.info('Assembled encoded corpus: %s in %s ms',
                 humanize.naturalsize(sys.getsizeof(self.encoded_corpus)),
                 humanize.intcomma(int((time.time() - start_time) * 1000)))
    self.corpus_length = len(self.encoded_corpus)
    self.sequence_length = self.training_opts.sequence_length
    if self.sequence_length >= self.corpus_length:
      max_sequence_length = self.corpus_length - 1
      raise errors.UserError(
          f'Requested training.sequence_length ({self.sequence_length}) is '
          f'larger than the corpus ({self.corpus_length}). '
          f'Reduce the sequence length to <= {max_sequence_length}.')
    self.batch_size = min(
        training_opts.batch_size,
        max(self.corpus_length - self.training_opts.sequence_length - 1, 1))
    if self.batch_size < training_opts.batch_size:
      logging.warning(
          'Requested training.batch_size (%d) is larger than the corpus (%d). '
          'Reduced batch size to %d', training_opts.batch_size,
          self.corpus_length, self.batch_size)

    # Set this publicly visibly attribute. The number of steps per epoch is
    # the total number of batches per epoch.
    self.steps_per_epoch = int(
        self.corpus_length / (self.batch_size * self.sequence_length))

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
    # _ = keras.utils.to_categorical(data.y, self.corpus.vocabulary_size)

    X = np.zeros((len(data.X), len(data.X[0]), self.corpus.vocabulary_size),
                 dtype=np.bool)
    y = np.zeros((len(data.y), self.corpus.vocabulary_size), dtype=np.bool)
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

  def __init__(self, corpus: corpuses.Corpus,
               training_opts: model_pb2.TrainingOptions):
    super(LazyVectorizingGenerator, self).__init__(corpus, training_opts)
    self.skip = 1  # TODO(cec): Add this as a field in Model.TrainingOptions.

    # Start index into the encoded corpus.
    self.i = 0

    # Create a dummy batch of data to get the size of it.
    x = np.zeros(
        (self.batch_size, self.sequence_length,
         self.corpus.vocabulary_size),
        dtype=np.bool)
    y = np.zeros((self.batch_size, self.corpus.vocabulary_size), dtype=np.bool)
    batch_size = sys.getsizeof(x) + sys.getsizeof(y)
    logging.info('%s: %s per batch, %s per epoch, %s total',
                 type(self).__name__,
                 humanize.naturalsize(batch_size),
                 humanize.naturalsize(batch_size * self.steps_per_epoch),
                 humanize.naturalsize(batch_size * self.steps_per_epoch *
                                      self.training_opts.num_epochs))

  def __next__(self) -> DataBatch:
    """Generate the next batch of X, y pairs."""
    start_time = time.time()
    # Return to the start of the encoded corpus if we've run out of text.
    if (self.i + self.batch_size + self.sequence_length + 1 >=
        self.corpus_length):
      self.i = 0
      if self.shuffle:
        self.encoded_corpus = self.corpus.GetTrainingData(shuffle=True)

    X_data = np.ndarray((self.batch_size, self.sequence_length),
                        dtype=np.int32)
    y_data = np.ndarray((self.batch_size,), dtype=np.int32)
    for i in range(0, self.batch_size, self.skip):
      sequence = np.array(
          self.encoded_corpus[self.i + i:self.i + i + self.sequence_length])
      next_token = self.encoded_corpus[self.i + i + self.sequence_length]
      X_data[i:, ] = sequence
      y_data[i] = next_token

    logging.debug('%s %dx%d batch %.2f ms',
                  type(self).__name__,
                  self.batch_size, self.sequence_length,
                  (time.time() - start_time) * 1000)

    self.i += self.batch_size
    return self.Vectorize(DataBatch(X=X_data, y=y_data))


class BatchesGenerator(DataGeneratorBase):

  def __init__(self, corpus: corpuses.Corpus,
               training_opts: model_pb2.TrainingOptions):
    super(BatchesGenerator, self).__init__(corpus, training_opts)

    # Index into the batches arrays.
    self.i = 0

    corpus_end = self.steps_per_epoch * self.batch_size * self.sequence_length

    x = np.reshape(
        self.encoded_corpus[:corpus_end],
        [self.batch_size, self.steps_per_epoch * self.sequence_length])

    y = np.reshape(
        self.encoded_corpus[1:corpus_end + 1],
        [self.batch_size, self.steps_per_epoch * self.sequence_length])
    # One hot encode the y data.
    y = np.eye(self.corpus.atomizer.vocab_size)[y]

    # TODO(cec):
    # x = self.encoded_corpus[corpus_end]
    # y = np.copy(self.encoded_corpus[corpus_end])
    # # Shift the y data along by one.
    # y[:-1] = x[1:]
    # # The end of the corpus wraps around to the start.
    # y[-1] = x[0]
    x_batches = np.split(x.reshape(self.batch_size, -1),
                         self.steps_per_epoch, 1)
    y_batches = np.split(y.reshape(self.batch_size, -1),
                         self.steps_per_epoch, 1)
    self.batches = [DataBatch(X=x, y=y) for x, y in zip(x_batches, y_batches)]

    batch_size = sys.getsizeof(self.batches[0])
    logging.info('%s: %s per batch, %s per epoch, %s total',
                 type(self).__name__,
                 humanize.naturalsize(batch_size),
                 humanize.naturalsize(batch_size * self.steps_per_epoch),
                 humanize.naturalsize(batch_size * self.steps_per_epoch *
                                      self.training_opts.num_epochs))

  def __next__(self) -> DataBatch:
    """Generate the next batch of X, y pairs."""
    self.i += 1
    # Wrap around to the start of the corpus if we have ran out of data.
    if self.i >= self.steps_per_epoch:
      self.i = 0
    return self.batches[self.i]


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
  return BatchesGenerator(corpus, training_opts)
