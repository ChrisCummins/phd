"""Utilities for LSTM models."""
# Quiet keras import, see https://stackoverflow.com/a/51567328
import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
import keras

sys.stderr = stderr

import tensorflow as tf

from labm8.py import app

app.DEFINE_boolean(
  "cudnn_lstm",
  True,
  "If set, use CuDNNLSTM implementation when a GPU is available. Else use "
  "default Keras implementation. Note that the two implementations are "
  "incompatible - a model saved using one LSTM type cannot be restored using "
  "the other LSTM type.",
)

FLAGS = app.FLAGS


def SetAllowedGrowthOnKerasSession():
  """Allow growth on GPU for Keras."""
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)
  return session


def LstmLayer(*args, **kwargs):
  """Construct an LSTM layer.

  If a GPU is available and --cudnn_lstm, this will use NVIDIA's fast CuDNNLSTM
  implementation. Else it will use Keras' builtin LSTM, which is much slower but
  works on CPU.
  """
  if FLAGS.cudnn_lstm and tf.compat.v1.test.is_gpu_available():
    return keras.layers.CuDNNLSTM(*args, **kwargs)
  else:
    return keras.layers.LSTM(*args, **kwargs, implementation=1)


def SegmentSumLayer(
  encoded_sequences,
  segment_ids,
  batch_size: int,
  max_sequence_length: int,
  max_output_sequence_length: int,
) -> keras.layers.Lambda:
  """Construct a layer which sum the encoded sequences by their segment IDs.

  Args:
    encoded_sequences: Shape (batch_size, max_sequence_length,
      embedding_dimensionality).
    segment_ids: Shape (batch_size, segment_ids).
    batch_size: The size of each batch.
    max_sequence_length: The maximum length of the input sequence.
    max_output_sequence_length: The length of the output sequence.

  Returns:
    A tensor of shape (batch_size, max_output_sequence_length, vocabulary_size).
  """

  def SegmentSum(args) -> tf.Tensor:
    """Compute the segment sums."""
    encoded_sequences, segment_ids = args

    segment_ids = tf.cast(segment_ids, dtype=tf.int32)

    # Perform a segment sum for each row in the batch independently.
    segment_sums = [
      tf.math.unsorted_segment_sum(
        data=encoded_sequences[i][:max_sequence_length],
        segment_ids=segment_ids[i][:max_sequence_length],
        num_segments=max_output_sequence_length,
      )
      for i in range(batch_size)
    ]

    return tf.stack(segment_sums, axis=0)

  return keras.layers.Lambda(SegmentSum)([encoded_sequences, segment_ids])


def SliceToSizeLayer(segmented_input, selector_vector) -> keras.layers.Lambda:
  """Slice the segmented_input shape to match the selector_vector
  dimensionality."""

  def SliceToSize(args) -> tf.Tensor:
    """Slice the input."""
    segmented_inputs, selector_vector = args
    max_number_nodes = tf.shape(selector_vector)[1]
    segmented_inputs = segmented_inputs[:, :max_number_nodes]
    return segmented_inputs

  return keras.layers.Lambda(SliceToSize)([segmented_input, selector_vector])
