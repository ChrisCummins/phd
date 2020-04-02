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
"""This module defines the abstract base class for LSTM models."""
import io
import pathlib
import tempfile
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional

import programl.ml.batch.batch_results
from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.seq import graph2seq
from labm8.py import app
from labm8.py import humanize
from labm8.py import progress
from programl.ml.batch import batch_data as batches
from programl.ml.epoch import epoch
from programl.ml.model import classifier_base
from programl.ml.model.lstm import lstm_utils as utils
from third_party.py.tensorflow import tf


FLAGS = app.FLAGS

app.DEFINE_integer(
  "lang_model_hidden_size",
  64,
  "The size of hidden layer(s) in the LSTM baselines.",
)
app.DEFINE_integer(
  "heuristic_model_hidden_size", 32, "The size of the dense output layer."
)
app.DEFINE_integer(
  "batch_size",
  64,
  "The number of padded sequences to concatenate into a batch.",
)
app.DEFINE_boolean(
  "cudnn_lstm",
  True,
  "If set, use CuDNNLSTM implementation when a GPU is available. Else use "
  "default Keras implementation. Note that the two implementations are "
  "incompatible - a model saved using one LSTM type cannot be restored using "
  "the other LSTM type.",
)


class LstmBase(classifier_base.ClassifierBase):
  """Abstract base class for LSTM models."""

  def __init__(
    self,
    *args,
    padded_sequence_length: Optional[int] = None,
    graph2seq_encoder: Optional[graph2seq.EncoderBase] = None,
    batch_size: Optional[int] = None,
    **kwargs,
  ):
    super(LstmBase, self).__init__(*args, **kwargs)

    self.batch_size = batch_size or FLAGS.batch_size

    # Determine the size of padded sequences. Use the requested
    # padded_sequence_length, or the maximum encoded length if it is shorter.
    self.padded_sequence_length = (
      padded_sequence_length or FLAGS.padded_sequence_length
    )

    self.encoder = graph2seq_encoder or self.GetEncoder()

    # After instantiating the encoder, see if we can reduce the padded sequence
    # length.
    self.padded_sequence_length = min(
      self.padded_sequence_length, self.encoder.max_encoded_length
    )

    # Reset any previous Tensorflow session. This is required when running
    # consecutive LSTM models in the same process.
    tf.compat.v1.keras.backend.clear_session()

    # Set by Initialize() and LoadModelData().
    self.session = None
    self.graph = None

  def MakeLstmLayer(self, *args, **kwargs):
    """Construct an LSTM layer.

    If a GPU is available and --cudnn_lstm, this will use NVIDIA's fast
    CuDNNLSTM implementation. Else it will use Keras' builtin LSTM, which is
    much slower but works on CPU.
    """
    if self.gpu and FLAGS.cudnn_lstm and tf.compat.v1.test.is_gpu_available():
      return tf.compat.v1.keras.layers.CuDNNLSTM(*args, **kwargs)
    else:
      return tf.compat.v1.keras.layers.LSTM(*args, **kwargs, implementation=1)

    return super(LstmBase, self).GraphReader(
      epoch_type=epoch_type,
      graph_db=graph_db,
      filters=filters,
      limit=limit,
      ctx=ctx,
    )

  def CreateModelData(self) -> None:
    """Initialize an LSTM model. This """
    # Create the Tensorflow session and graph for the model.
    self.session = utils.SetAllowedGrowthOnKerasSession()
    self.graph = tf.compat.v1.get_default_graph()

    # To enable thread-safe use of a Keras model we must make sure to fix the
    # graph and session whenever we are going to use self.model.
    with self.graph.as_default():
      tf.compat.v1.keras.backend.set_session(self.session)
      self.model = self.CreateKerasModel()

    self.FinalizeKerasModel()

  def FinalizeKerasModel(self) -> None:
    """Finalize a newly instantiated keras model.

    To enable thread-safe use of the Keras model we must ensure that the
    computation graph is fully instantiated from the master thread before the
    first call to RunBatch(). Keras lazily instantiates parts of the graph
    which we can force by performing the necessary ops now:
      * training ops: make sure those are created by running the training loop
        on a small batch of data.
      * save/restore ops: make sure those are created by running save_model()
        and throwing away the generated file.

    Once we have performed those actions, we can freeze the computation graph
    to make explicit the fact that later operations are not permitted to modify
    the graph.
    """
    with self.graph.as_default():
      tf.compat.v1.keras.backend.set_session(self.session)
      # To enable thread-safe use of the Keras model we must ensure that
      # the computation graph is fully instantiated before the first call
      # to RunBatch(). Keras lazily instantiates parts of the graph (such as
      # training ops), so make sure those are created by running the training
      # loop now on a single graph.
      reader = graph_database_reader.BufferedGraphReader(
        self.graph_db, limit=self.warm_up_batch_size
      )
      batch = self.MakeBatch(epoch.EpochType.TRAIN, reader)
      assert batch.graph_count == self.warm_up_batch_size
      self.RunBatch(epoch.EpochType.TRAIN, batch)

      # Run private model methods that instantiate graph components.
      # See: https://stackoverflow.com/a/46801607
      self.model._make_predict_function()
      self.model._make_test_function()
      self.model._make_train_function()

      # Saving the graph also creates new ops, so run it now.
      with tempfile.TemporaryDirectory(prefix="ml4pl_lstm_") as d:
        self.model.save(pathlib.Path(d) / "delete_md.h5")

    # Finally we have instantiated the graph, so freeze it to mane any
    # implicit modification raise an error.
    self.graph.finalize()

  @property
  def warm_up_batch_size(self) -> int:
    """Get the size of the batch used for warm-up runs of the LSTM model."""
    return 1

  def GetEncoder(self) -> graph2seq.EncoderBase:
    """Construct the graph encoder."""
    raise NotImplementedError("abstract class")

  def Summary(self) -> str:
    """Get a summary"""
    buf = io.StringIO()
    self.model.summary(print_fn=lambda msg: print(msg, file=buf))
    print(
      "Using padded sequence length "
      f"{humanize.DecimalPrefix(self.padded_sequence_length, '')} from maximum "
      f"{humanize.DecimalPrefix(self.encoder.max_encoded_length, '')} "
      f"(max {(1 - (self.padded_sequence_length / self.encoder.max_encoded_length)):.3%} "
      "sequence truncation)",
      file=buf,
    )
    return buf.getvalue()

  @property
  def padded_vocabulary_size(self) -> int:
    return self.encoder.vocabulary_size + 1

  @property
  def padding_element(self) -> int:
    return self.encoder.vocabulary_size

  def GetBatchOfGraphs(
    self, graph_iterator: Iterable[graph_tuple_database.GraphTuple]
  ) -> List[graph_tuple_database.GraphTuple]:
    """Read a list of <= batch_size graphs from a graph_iterator."""
    # Peel off a batch of graphs to process.
    graphs: List[graph_tuple_database.GraphTuple] = []
    while len(graphs) < self.batch_size:
      try:
        graph = next(graph_iterator)
      except StopIteration:
        # We have run out of graphs.
        break
      graphs.append(graph)
    return graphs

  def GetModelData(self) -> Any:
    """Get the model state."""
    # According to https://keras.io/getting-started/faq/, it is not recommended
    # to pickle a Keras model. So as a workaround, I use Keras's saving
    # mechanism to store the weights, and pickle that.
    with tempfile.TemporaryDirectory(prefix="lstm_pickle_") as d:
      path = pathlib.Path(d) / "weights.h5"
      with self.graph.as_default():
        tf.compat.v1.keras.backend.set_session(self.session)
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
      if self.session:
        self.session.close()
      tf.compat.v1.reset_default_graph()
      self.session = utils.SetAllowedGrowthOnKerasSession()
      self.graph = tf.compat.v1.get_default_graph()

      with self.graph.as_default():
        tf.compat.v1.keras.backend.set_session(self.session)
        self.model = tf.compat.v1.keras.models.load_model(path)

    self.FinalizeKerasModel()

  def CreateKerasModel(self) -> tf.compat.v1.keras.Model:
    """Create the LSTM model."""
    raise NotImplementedError("abstract class")

  def RunBatch(
    self,
    epoch_type: epoch.EpochType,
    batch: batches.BatchData,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> programl.ml.batch.batch_results.BatchResults:
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
    # We can only get the loss on training.
    loss = None

    with self.graph.as_default():
      tf.compat.v1.keras.backend.set_session(self.session)

      if epoch_type == epoch.EpochType.TRAIN:
        loss, *_ = self.model.train_on_batch(
          batch.model_data.x, batch.model_data.y
        )

      predictions = self.model.predict_on_batch(batch.model_data.x)

    return programl.ml.batch.batch_results.BatchResults.Create(
      targets=batch.model_data.targets,
      predictions=batch.model_data.GetPredictions(predictions, ctx=ctx),
      loss=loss,
    )
