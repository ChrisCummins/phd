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
"""An LSTM for instruction classification."""
import pathlib
import tempfile
from typing import Any
from typing import Dict

import numpy as np

from labm8.py import app
from labm8.py.progress import NullContext
from labm8.py.progress import ProgressContext
from programl.ml.batch.batch_data import BatchData
from programl.ml.batch.batch_results import BatchResults
from programl.ml.model.lstm.lstm_batch import LstmBatchData
from programl.ml.model.model import Model
from programl.proto import epoch_pb2
from third_party.py.tensorflow import tf


FLAGS = app.FLAGS

app.DEFINE_integer(
  "hidden_size", 64, "The size of hidden layer(s).",
)
app.DEFINE_integer(
  "batch_size",
  256,
  "The number of padded sequences to concatenate into a batch.",
)
app.DEFINE_integer(
  "padded_sequence_length",
  5000,
  "For node-level models, the padded/truncated length of encoded node "
  "sequences.",
)
app.DEFINE_boolean(
  "cudnn_lstm",
  True,
  "If set, use CuDNNLSTM implementation when a GPU is available. Else use "
  "default Keras implementation. Note that the two implementations are "
  "incompatible - a model saved using one LSTM type cannot be restored using "
  "the other LSTM type.",
)


class Lstm(Model):
  """An LSTM model for node-level classification."""

  def __init__(
    self,
    vocabulary: Dict[str, int],
    node_y_dimensionality: int,
    test_only: bool = False,
    name: str = "lstm",
  ):
    """Constructor."""
    super(Lstm, self).__init__(
      test_only=test_only, vocabulary=vocabulary, name=name
    )

    self.vocabulary = vocabulary
    self.node_y_dimensionality = node_y_dimensionality

    # Flag values.
    self.batch_size = FLAGS.batch_size
    self.padded_sequence_length = FLAGS.padded_sequence_length

    # Reset any previous Tensorflow session. This is required when running
    # consecutive LSTM models in the same process.
    tf.compat.v1.keras.backend.clear_session()

  @staticmethod
  def MakeLstmLayer(*args, **kwargs):
    """Construct an LSTM layer.

    If a GPU is available and --cudnn_lstm, this will use NVIDIA's fast
    CuDNNLSTM implementation. Else it will use Keras' builtin LSTM, which is
    much slower but works on CPU.
    """
    if FLAGS.cudnn_lstm and tf.compat.v1.test.is_gpu_available():
      return tf.compat.v1.keras.layers.CuDNNLSTM(*args, **kwargs)
    else:
      return tf.compat.v1.keras.layers.LSTM(*args, **kwargs, implementation=1)

  def CreateKerasModel(self) -> tf.compat.v1.keras.Model:
    """Construct the tensorflow computation graph."""
    sequence_input = tf.compat.v1.keras.layers.Input(
      batch_shape=(self.batch_size, self.padded_sequence_length,),
      dtype="int32",
      name="sequence_in",
    )
    embedded_inputs = tf.compat.v1.keras.layers.Embedding(
      input_dim=len(self.vocabulary) + 2,
      input_length=self.padded_sequence_length,
      output_dim=FLAGS.hidden_size,
      name="embedding",
    )(sequence_input)

    selector_vector = tf.compat.v1.keras.layers.Input(
      batch_shape=(self.batch_size, self.padded_sequence_length, 2),
      dtype="float32",
      name="selector_vector",
    )

    lang_model_input = tf.compat.v1.keras.layers.Concatenate(
      axis=2, name="embedded_inputs_and_selector_vectors"
    )([embedded_inputs, selector_vector],)

    # Recurrent layers.
    lang_model = self.MakeLstmLayer(
      FLAGS.hidden_size, return_sequences=True, name="lstm_1"
    )(lang_model_input)
    lang_model = self.MakeLstmLayer(
      FLAGS.hidden_size,
      return_sequences=True,
      return_state=False,
      name="lstm_2",
    )(lang_model)

    # Dense layers.
    lang_model = tf.compat.v1.keras.layers.Dense(
      FLAGS.hidden_size, activation="relu", name="dense_1",
    )(lang_model)
    node_out = tf.compat.v1.keras.layers.Dense(
      self.node_y_dimensionality, activation="sigmoid", name="node_out",
    )(lang_model)

    model = tf.compat.v1.keras.Model(
      inputs=[sequence_input, selector_vector], outputs=[node_out],
    )
    model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy"],
      loss_weights=[1.0],
    )

    return model

  def CreateModelData(self, test_only: bool) -> None:
    """Initialize an LSTM model. This is called during Initialize()."""
    # Create the Tensorflow session and graph for the model.
    tf.get_logger().setLevel("ERROR")
    SetAllowedGrowthOnKerasSession()
    self.model = self.CreateKerasModel()

  def RunBatch(
    self,
    epoch_type: epoch_pb2.EpochType,
    batch_data: BatchData,
    ctx: ProgressContext = NullContext,
  ) -> BatchResults:
    """Run a batch of data through the model.

    Args:
      epoch_type: The type of the current epoch.
      batch: A batch of graphs and model data. This requires that batch data has
        'x' and 'y' properties that return lists of model inputs, a `targets`
        property that returns a flattened list of targets, a `GetPredictions()`
        method that recieves as input the data generated by model and returns
        a flattened array of the same shape as `targets`.
      ctx: A logging context.
    """
    model_data: LstmBatchData = batch_data.model_data

    x = [model_data.encoded_sequences, model_data.selector_vectors]

    if epoch_type == epoch_pb2.TRAIN:
      loss, *_ = self.model.train_on_batch(x, [model_data.node_labels])
    else:
      loss = None

    padded_predictions = self.model.predict_on_batch(x)

    # Reshape the outputs.
    predictions = self.ReshapePaddedModelOutput(model_data, padded_predictions)

    # Flatten the targets and predictions lists.
    # Shape (batch_node_count, node_y_dimensionality).
    targets = np.concatenate(model_data.targets)
    predictions = np.concatenate(predictions)

    return BatchResults.Create(
      targets=targets, predictions=predictions, loss=loss,
    )

  def ReshapePaddedModelOutput(
    self, model_data: LstmBatchData, padded_outputs
  ) -> np.array:
    """Reshape the model outputs to an array of predictions of same shape as
    targets. Zeros are used as padding values when the model produces fewer
    outputs than there are nodes in a graph.
    """
    if padded_outputs.shape != (
      self.batch_size,
      self.padded_sequence_length,
      self.node_y_dimensionality,
    ):
      raise TypeError(
        f"Model produced output with shape {padded_outputs.shape} but "
        f"expected outputs with shape ({self.batch_size}, {self.padded_sequence_length})"
      )
    outputs = []
    for graph_node_size, padded_output in zip(
      model_data.graph_node_sizes,
      padded_outputs[: len(model_data.graph_node_sizes)],
    ):
      active_nodes = padded_output[:graph_node_size]
      padding = np.zeros(
        (
          max(graph_node_size - self.padded_sequence_length, 0),
          self.node_y_dimensionality,
        ),
        dtype=np.float32,
      )
      outputs.append(np.concatenate((active_nodes, padding)))
    return outputs

  def GetModelData(self) -> Any:
    """Get the model state."""
    # According to https://keras.io/getting-started/faq/, it is not recommended
    # to pickle a Keras model. So as a workaround, I use Keras's saving
    # mechanism to store the weights, and pickle that.
    with tempfile.TemporaryDirectory(prefix="lstm_pickle_") as d:
      path = pathlib.Path(d) / "weights.h5"
      self.model.save(path)
      with open(path, "rb") as f:
        model_data = f.read()
    return model_data

  def LoadModelData(self, data_to_load: Any) -> None:
    """Restore the model state."""
    # Load the weights from a file generated by ModelDataToSave().
    with tempfile.TemporaryDirectory(prefix="lstm_pickle_") as d:
      path = pathlib.Path(d) / "weights.h5"
      with open(path, "wb") as f:
        f.write(data_to_load)

      # The default TF graph is finalized in Initialize(), so we must
      # first reset the session and create a new graph.
      tf.compat.v1.reset_default_graph()
      SetAllowedGrowthOnKerasSession()

      self.model = tf.compat.v1.keras.models.load_model(path)


def SetAllowedGrowthOnKerasSession():
  """Allow growth on GPU for Keras."""
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(session)
  return session
