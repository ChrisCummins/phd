# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for LSTM models."""
from labm8.py import app
from third_party.py.tensorflow import tf

FLAGS = app.FLAGS


def SetAllowedGrowthOnKerasSession():
  """Allow growth on GPU for Keras."""
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)
  return session


def SegmentSumLayer(
  encoded_sequences,
  segment_ids,
  batch_size: int,
  max_sequence_length: int,
  max_output_sequence_length: int,
) -> tf.compat.v1.keras.layers.Lambda:
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

  return tf.compat.v1.keras.layers.Lambda(SegmentSum)(
    [encoded_sequences, segment_ids]
  )


def SliceToSizeLayer(
  segmented_input, selector_vector
) -> tf.compat.v1.keras.layers.Lambda:
  """Slice the segmented_input shape to match the selector_vector
  dimensionality."""

  def SliceToSize(args) -> tf.Tensor:
    """Slice the input."""
    segmented_inputs, selector_vector = args
    max_number_nodes = tf.shape(selector_vector)[1]
    segmented_inputs = segmented_inputs[:, :max_number_nodes]
    return segmented_inputs

  return tf.compat.v1.keras.layers.Lambda(SliceToSize)(
    [segmented_input, selector_vector]
  )
