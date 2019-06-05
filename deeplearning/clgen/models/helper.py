import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops


class CustomInferenceHelper(seq2seq.TrainingHelper):
  """An inference helper that takes a seed text"""

  def __init__(self, input_seed, sequence_length, seed_length, embedding,
               temperature):
    super(CustomInferenceHelper, self).__init__(inputs=input_seed,
                                                sequence_length=sequence_length,
                                                time_major=False)

    self._xlate = embedding
    self._seed_length = seed_length
    self._softmax_temperature = temperature

  def initialize(self, name=None):
    return super(CustomInferenceHelper, self).initialize(name=name)

  def sample(self, time, outputs, state, name=None):
    if self._softmax_temperature is not None:
      outputs = outputs / self._softmax_temperature

    sampler = categorical.Categorical(logits=outputs)
    sample_ids = sampler.sample()
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with tf.name_scope(name, "CIHNextInputs", [time, outputs, state]):
      next_time = time + 1
      finished = (next_time >= self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      seed_done = (next_time >= self._seed_length)

      def read_from_ta(inp):
        return inp.read(next_time)

      next_inputs = tf.case(
          [(all_finished, lambda: self._zero_inputs),
           (tf.logical_not(seed_done),
            lambda: nest.map_structure(read_from_ta, self._input_tas))],
          default=lambda: tf.stop_gradient(
              tf.nn.embedding_lookup(self._xlate, sample_ids)))
      return (finished, next_inputs, state)
