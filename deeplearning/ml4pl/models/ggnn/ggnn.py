# Copyright 2019 the ProGraML authors.
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
"""A gated graph neural network classifier."""
import typing
from typing import Callable
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np
import torch
from torch import nn

from deeplearning.ml4pl.graphs.labelled import graph_batcher
from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.llvm2graph import node_encoder
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import run
from deeplearning.ml4pl.models.ggnn.ggnn_config import GGNNConfig
from deeplearning.ml4pl.models.ggnn.ggnn_modules import GGNNModel
from labm8.py import app
from labm8.py import progress

FLAGS = app.FLAGS

app.DEFINE_float(
  "label_conv_threshold",
  0.995,
  "convergence interval: fraction of labels that need to be stable",
)
app.DEFINE_integer(
  "label_conv_stable_steps",
  1,
  "required number of consecutive steps within the convergence interval",
)
app.DEFINE_integer(
  "label_conv_max_timesteps",
  1000,
  "The maximum number of iterations to attempt to reach label convergence. "
  "No effect when --unroll_strategy is not label_convergence.",
)

app.DEFINE_list(
  "layer_timesteps",
  # 4x[2] is a good default!
  ["2", "2", "2", "2"],
  "A list of layers, and the number of steps for each layer.",
)
# 0.00025 is a good default!
app.DEFINE_float("learning_rate", 0.00025, "The initial learning rate.")

# not clamping, i.e. setting to 0.0 is a good default.
app.DEFINE_float("clamp_gradient_norm", 0.0, "Clip gradients to L-2 norm.")


app.DEFINE_integer("hidden_size", 200, "The size of hidden layer(s).")
app.DEFINE_string(
  "inst2vec_embeddings",
  "random",
  "The type of per-node inst2vec embeddings to use. One of {zero, constant, random, random_const, finetune, none}"
  "'none' maps all statements to a single token, so overall there exist only !ID and a !STMT tokens.",
)
app.DEFINE_string(
  "unroll_strategy",
  "none",
  "The unroll strategy to use. One of: "
  "{none, constant, edge_count, data_flow_max_steps, label_convergence} "
  "constant: Unroll by a constant number of steps. The total number of steps is "
  "defined in FLAGS.test_layer_timesteps",
)

app.DEFINE_list(
  "test_layer_timesteps",
  ["0"],
  "Set when unroll_strategy is 'constant'. Assumes that the length <= len(layer_timesteps)."
  "Unrolls the GGNN proper for a fixed number of timesteps during eval().",
)

app.DEFINE_boolean(
  "limit_max_data_flow_steps_during_training",
  True,
  "If set, limit the size of dataflow-annotated graphs used to train and "
  "validate models to only those with data_flow_steps <= sum(layer_timesteps). "
  "This has no effect for graph databases with no dataflow annotations, or "
  "for testing epochs.",
)
# We assume that position_embeddings exist in every dataset.
# the flag now only controls whether they are used or not.
# This could be nice for ablating our model and also debugging with and without.

app.DEFINE_boolean(
  "position_embeddings",
  # False shall be a good default for small datasets.
  True,
  "Whether to use position embeddings as signals for edge order."
  "We expect them to be part of the ds anyway, but you can toggle off their effect.",
)

app.DEFINE_boolean("use_edge_bias", True, "")

app.DEFINE_boolean(
  "msg_mean_aggregation",
  True,
  "If true, normalize incoming messages by the number of incoming messages.",
)
app.DEFINE_float(
  # 0.2 is a good default
  "graph_state_dropout",
  0.2,
  "Graph state dropout rate.",
)
app.DEFINE_float(
  # 0.0 is a good default
  "edge_weight_dropout",
  0.0,
  "Edge weight dropout rate.",
)
app.DEFINE_float(
  # 0.0 is a good default, found without aux input.
  "output_layer_dropout",
  0.0,
  "Dropout rate on the output layer.",
)
app.DEFINE_float(
  "intermediate_loss_weight",
  0.2,
  "The actual loss is computed as loss + factor * intermediate_loss",
)
app.DEFINE_integer(
  "aux_in_layer_size",
  32,
  "Size for MLP that combines graph_features and aux_in features",
)
app.DEFINE_boolean(
  "log1p_graph_x",
  True,
  "If set, apply a log(x + 1) transformation to incoming auxiliary graph-level features.",
)


####### DEBBUGING HELPERS ##########################
DEBUG = False


def assert_no_nan(tensor_list):
  for i, t in enumerate(tensor_list):
    assert not torch.isnan(t).any(), f"{i}: {tensor_list}"


def nan_hook(self, inp, output):
  """Checks return values of any forward() function for NaN"""
  if not isinstance(output, tuple):
    outputs = [output]
  else:
    outputs = output

  for i, out in enumerate(outputs):
    nan_mask = torch.isnan(out)
    if nan_mask.any():
      print("In", self.__class__.__name__)
      raise RuntimeError(
        f"Found NAN in output {i} at indices: ",
        nan_mask.nonzero(),
        "where:",
        out[nan_mask.nonzero()[:, 0].unique(sorted=True)],
      )


##########################################


class GgnnBatchData(NamedTuple):
  """The model-specific data generated for a batch."""

  # A combination of one or more graphs into a single disconnected graph.
  disjoint_graph: graph_tuple.GraphTuple
  # A list of graphs that were used to construct the disjoint graph.
  graphs: List[graph_tuple_database.GraphTuple]


class Ggnn(classifier_base.ClassifierBase):
  """A gated graph neural network."""

  def __init__(self, *args, **kwargs):
    """Constructor."""
    super(Ggnn, self).__init__(*args, **kwargs)

    # set some global config values

    # Instantiate model
    config = GGNNConfig(
      num_classes=self.y_dimensionality,
      has_graph_labels=self.graph_db.graph_y_dimensionality > 0,
      has_aux_input=self.graph_db.graph_x_dimensionality > 0,
    )

    inst2vec_embeddings = node_encoder.GraphNodeEncoder().embeddings_tables[0]
    inst2vec_embeddings = torch.from_numpy(
      np.array(inst2vec_embeddings, dtype=np.float32)
    )
    self.model = GGNNModel(
      config,
      pretrained_embeddings=inst2vec_embeddings,
      test_only=FLAGS.test_only,
    )
    app.Log(
      1,
      "Using device %s with dtype %s",
      self.model.dev,
      torch.get_default_dtype(),
    )

    if DEBUG:
      for submodule in self.model.modules():
        submodule.register_forward_hook(nan_hook)

    self.model.to(self.model.dev)

  def MakeBatch(
    self,
    epoch_type: epoch.Type,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch of data from an iterator of graphs.

    Returns:
      A single batch of data for feeding into RunBatch(). A batch consists of a
      list of graph IDs and a model-defined blob of data. If the list of graph
      IDs is empty, the batch is discarded and not fed into RunBatch().
    """
    # TODO(github.com/ChrisCummins/ProGraML/issues/24): The new graph batcher
    # implementation is not well suited for reading the graph IDs, hence this
    # somewhat clumsy iterator wrapper. A neater approach would be to create
    # a graph batcher which returns a list of graphs in the batch.
    class GraphIterator(object):
      """A wrapper around a graph iterator which records graph IDs."""

      def __init__(self, graphs: Iterable[graph_tuple_database.GraphTuple]):
        self.input_graphs = graphs
        self.graphs_read: List[graph_tuple_database.GraphTuple] = []

      def __iter__(self):
        return self

      def __next__(self):
        graph: graph_tuple_database.GraphTuple = next(self.input_graphs)
        self.graphs_read.append(graph)
        return graph.tuple

    graph_iterator = GraphIterator(graphs)

    # Create a disjoint graph out of one or more input graphs.
    batcher = graph_batcher.GraphBatcher.CreateFromFlags(
      graph_iterator, ctx=ctx
    )

    try:
      disjoint_graph = next(batcher)
    except StopIteration:
      # We have run out of graphs.
      return batches.EndOfBatches()

    # Workaround for the fact that graph batcher may read one more graph than
    # actually gets included in the batch.
    if batcher.last_graph:
      graphs = graph_iterator.graphs_read[:-1]
    else:
      graphs = graph_iterator.graphs_read

    # Discard single-graph batches during training when there are graph
    # features. This is because we use batch normalization on incoming features,
    # and batch normalization requires > 1 items to normalize.
    if (
      len(graphs) <= 1
      and epoch_type == epoch.Type.TRAIN
      and disjoint_graph.graph_x_dimensionality
    ):
      return batches.EmptyBatch()

    return batches.Data(
      graph_ids=[graph.id for graph in graphs],
      data=GgnnBatchData(disjoint_graph=disjoint_graph, graphs=graphs),
    )

  def GraphReader(
    self,
    epoch_type: epoch.Type,
    graph_db: graph_tuple_database.Database,
    filters: Optional[List[Callable[[], bool]]] = None,
    limit: Optional[int] = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> graph_database_reader.BufferedGraphReader:
    """Construct a buffered graph reader.

    Args:
      epoch_type: The type of graph reader to return a graph reader for.
      graph_db: The graph database to read graphs from.
      filters: A list of filters to impose on the graph database reader.
      limit: The maximum number of rows to read.
      ctx: A logging context.

    Returns:
      A buffered graph reader instance.
    """
    filters = filters or []

    # Only read graphs with data_flow_steps <= message_passing_step_count if
    # --limit_max_data_flow_steps_during_training is set and we are not
    # in a test epoch.
    if (
      FLAGS.limit_max_data_flow_steps_during_training
      and self.graph_db.has_data_flow
      and (epoch_type == epoch.Type.TRAIN or epoch_type == epoch.Type.VAL)
    ):
      filters.append(
        lambda: graph_tuple_database.GraphTuple.data_flow_steps
        <= self.message_passing_step_count
      )

    # If we are batching my maximum node count and skipping graphs that are
    # larger than this, we can apply that filter to the SQL query now, rather
    # than reading the graphs and ignoring them later. This ensures that when
    # --max_{train,val}_per_epoch is set, the number of graphs that get used
    # matches the limit.
    if (
      FLAGS.graph_batch_node_count
      and FLAGS.max_node_count_limit_handler == "skip"
    ):
      filters.append(
        lambda: (
          graph_tuple_database.GraphTuple.node_count
          <= FLAGS.graph_batch_node_count
        )
      )

    return super(Ggnn, self).GraphReader(
      epoch_type=epoch_type,
      graph_db=graph_db,
      filters=filters,
      limit=limit,
      ctx=ctx,
    )

  @property
  def message_passing_step_count(self) -> int:
    return self.layer_timesteps.sum()

  @property
  def layer_timesteps(self) -> np.array:
    return np.array([int(x) for x in FLAGS.layer_timesteps])

  def get_unroll_steps(
    self, epoch_type: epoch.Type, batch: batches.Data, unroll_strategy: str,
  ) -> int:
    """Determine the unroll factor from the --unroll_strategy flag, and the batch log."""
    # Determine the unrolling strategy.
    if unroll_strategy == "none":
      # Perform no unrolling. The inputs are processed according to layer_timesteps
      return 0
    elif unroll_strategy == "constant":
      # Unroll by a constant number of steps according to test_layer_timesteps
      return 0
    elif unroll_strategy == "data_flow_max_steps":
      max_data_flow_steps = max(
        graph.data_flow_steps for graph in batch.data.graphs
      )
      app.Log(3, "Determined max data flow steps to be %d", max_data_flow_steps)
      return max_data_flow_steps
    elif unroll_strategy == "edge_count":
      max_edge_count = max(graph.edge_count for graph in batch.data.graphs)
      app.Log(3, "Determined max edge count to be %d", max_edge_count)
      return max_edge_count
    elif unroll_strategy == "label_convergence":
      return 0
    else:
      raise app.UsageError(f"Unknown unroll strategy '{unroll_strategy}'")

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batches.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Results:
    disjoint_graph: graph_tuple.GraphTuple = batch.data.disjoint_graph

    # Batch to model-inputs
    # torch.from_numpy() shares memory with numpy!
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): maybe we can save
    # memory copies in the training loop if we can turn the data into the
    # required types (np.int64 and np.float32) once they come off the network
    # from the database, where smaller i/o size (int32) is more important.
    with ctx.Profile(5, "Sent data to GPU"):
      vocab_ids = torch.from_numpy(disjoint_graph.node_x[:, 0]).to(
        self.model.dev, torch.long
      )
      selector_ids = torch.from_numpy(disjoint_graph.node_x[:, 1]).to(
        self.model.dev, torch.long
      )
      # we need those as a result on cpu and can save device i/o
      cpu_labels = (
        disjoint_graph.node_y
        if disjoint_graph.has_node_y
        else disjoint_graph.graph_y
      )
      labels = torch.from_numpy(cpu_labels).to(self.model.dev)
      edge_lists = [
        torch.from_numpy(x).to(self.model.dev, torch.long)
        for x in disjoint_graph.adjacencies
      ]

      edge_positions = [
        torch.from_numpy(x).to(self.model.dev, torch.long)
        for x in disjoint_graph.edge_positions
      ]

    model_inputs = {
      "vocab_ids": vocab_ids,
      "selector_ids": selector_ids,
      "labels": labels,
      "edge_lists": edge_lists,
      "pos_lists": edge_positions,
    }

    # maybe fetch more inputs.
    if disjoint_graph.has_graph_y:
      assert (
        epoch_type != epoch.Type.TRAIN
        or disjoint_graph.disjoint_graph_count > 1
      ), f"graph_count is {disjoint_graph.disjoint_graph_count}"
      num_graphs = torch.tensor(disjoint_graph.disjoint_graph_count).to(
        self.model.dev, torch.long
      )
      graph_nodes_list = torch.from_numpy(
        disjoint_graph.disjoint_nodes_list
      ).to(self.model.dev, torch.long)

      aux_in = torch.from_numpy(disjoint_graph.graph_x).to(
        self.model.dev, torch.get_default_dtype()
      )
      model_inputs.update(
        {
          "num_graphs": num_graphs,
          "graph_nodes_list": graph_nodes_list,
          "aux_in": aux_in,
        }
      )

    # maybe calculate manual timesteps
    if epoch_type != epoch.Type.TRAIN and FLAGS.unroll_strategy in [
      "constant",
      "edge_count",
      "data_flow_max_steps",
      "label_convergence",
    ]:
      time_steps_cpu = np.array(
        self.get_unroll_steps(epoch_type, batch, FLAGS.unroll_strategy),
        dtype=np.int64,
      )
      time_steps_gpu = torch.from_numpy(time_steps_cpu).to(self.model.dev)
    else:
      time_steps_cpu = 0
      time_steps_gpu = None

    # RUN MODEL FORWARD PASS
    # enter correct mode of model
    if epoch_type == epoch.Type.TRAIN:
      if not self.model.training:
        self.model.train()
      outputs = self.model(**model_inputs, test_time_steps=time_steps_gpu)
    else:  # not TRAIN
      if self.model.training:
        self.model.eval()
        self.model.opt.zero_grad()
      with torch.no_grad():  # don't trace computation graph!
        outputs = self.model(**model_inputs, test_time_steps=time_steps_gpu)

    (
      logits,
      accuracy,
      logits,
      correct,
      targets,
      graph_features,
      *unroll_stats,
    ) = outputs

    loss = self.model.loss((logits, graph_features), targets)

    if epoch_type == epoch.Type.TRAIN:
      loss.backward()
      # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Clip gradients
      # (done). NB, pytorch clips by norm of the gradient of the model, while
      # tf clips by norm of the grad of each tensor separately. Therefore we
      # change default from 1.0 to 6.0.
      # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Anyway: Gradients
      # shouldn't really be clipped if not necessary?
      if self.model.config.clip_grad_norm > 0.0:
        nn.utils.clip_grad_norm_(
          self.model.parameters(), self.model.config.clip_grad_norm
        )
      self.model.opt.step()
      self.model.opt.zero_grad()

    # tg = targets.numpy()
    # tg = np.vstack(((tg + 1) % 2, tg)).T
    # assert np.all(labels.numpy() == tg), f"labels sanity check failed: labels={labels.numpy()},  tg={tg}"

    # TODO(github.com/ChrisCummins/ProGraML/issues/27): Learning rate schedule
    # will change this value.
    learning_rate = self.model.config.lr

    model_converged = unroll_stats[1] if unroll_stats else False
    iteration_count = unroll_stats[0] if unroll_stats else time_steps_cpu

    loss_value = loss.item()
    assert not np.isnan(loss_value), loss
    return batches.Results.Create(
      targets=cpu_labels,
      predictions=logits.detach().cpu().numpy(),
      model_converged=model_converged,
      learning_rate=learning_rate,
      iteration_count=iteration_count,
      loss=loss_value,
    )

  def GetModelData(self) -> typing.Any:
    return {
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.model.opt.state_dict(),
    }

  def LoadModelData(self, data_to_load: typing.Any) -> None:
    self.model.load_state_dict(data_to_load["model_state_dict"])
    # only restore opt if needed. opt should be None o/w.
    if not FLAGS.test_only:
      self.model.opt.load_state_dict(data_to_load["optimizer_state_dict"])


def main():
  """Main entry point."""
  run.Run(Ggnn)


if __name__ == "__main__":
  app.Run(main)
