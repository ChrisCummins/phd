"""Utilities for LSTM models."""
import keras
import tensorflow as tf

from labm8.py import app

app.DEFINE_boolean(
  "cudnn_lstm",
  False,
  "If set, use CuDNNLSTM implementation. Else use default Keras implementation",
)

FLAGS = app.FLAGS


def SetAllowedGrowthOnKerasSession():
  """Allow growth on GPU for Keras."""
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)


def LstmLayer(*args, **kwargs):
  """Construct an LSTM layer"""
  if FLAGS.cudnn_lstm:
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

  Returns:
    A tensor of shape (batch_size, max_output_sequence_length, vocabulary_size).
  """

  def SegmentSum(args) -> tf.Tensor:
    """Compute the segment sums."""
    encoded_sequences, segment_ids = args

    segment_ids = tf.cast(segment_ids, dtype=tf.int32)

    # Determine the number of segments. Segment IDs are assumed to be
    # zero-based in the range [0, ... n - 1].
    num_segments = tf.cast(tf.math.reduce_max(segment_ids) + 1, dtype=tf.int32)

    # Perform a segment sum for each row in the batch independently.
    segment_sums = [
      # Note the slice so that graphs larger than max_sequence_length are
      # truncated.
      tf.math.unsorted_segment_sum(
        data=encoded_sequences[i][:max_sequence_length],
        segment_ids=segment_ids[i][:max_sequence_length],
        num_segments=num_segments,
      )[:max_output_sequence_length]
      for i in range(batch_size)
    ]

    return tf.stack(segment_sums, axis=0)

  return keras.layers.Lambda(SegmentSum)([encoded_sequences, segment_ids])


def SliceToSizeLayer(segmented_input, selector_vector) -> keras.layers.Lambda:
  """Slice the segmented_input shape to match the selector_vector
  dimensionality."""

  def SliceToSize(args) -> tf.Tensor:
    """Slize the input."""
    segmented_inputs, selector_vector = args
    max_number_nodes = tf.shape(selector_vector)[1]
    segmented_inputs = segmented_inputs[:, :max_number_nodes]
    return segmented_inputs

  return keras.layers.Lambda(SliceToSize)([segmented_input, selector_vector])
