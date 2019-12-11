"""A gated graph neural network classifier."""
import math
import typing
from typing import Iterable
from typing import List
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from deeplearning.ml4pl.graphs.labelled import graph_batcher
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import node_encoder
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import run
from deeplearning.ml4pl.models.ggnn.ggnn_config import GGNNConfig
from deeplearning.ml4pl.models.ggnn.ggnn_modules import GGNNModel
from labm8.py import app
from labm8.py import progress

FLAGS = app.FLAGS

app.DEFINE_list(
  "layer_timesteps",
  ["2", "2", "2"],
  "A list of layers, and the number of steps for each layer.",
)
app.DEFINE_float("learning_rate", 0.001, "The initial learning rate.")


app.DEFINE_float("clamp_gradient_norm", 6.0, "Clip gradients to L-2 norm.")


app.DEFINE_integer("hidden_size", 200, "The size of hidden layer(s).")
app.DEFINE_string(
  "inst2vec_embeddings",
  "random",
  "The type of per-node inst2vec embeddings to use. One of: "
  "{constant,constant_zero,constant_random,finetune,random}.",
)
app.DEFINE_string(
  "unroll_strategy",
  "none",
  "The unroll strategy to use. One of: "
  "{none, constant, edge_count, data_flow_max_steps, label_convergence} "
  "constant: Unroll by a constant number of steps. The total number of steps is "
  "(unroll_factor * message_passing_step_count).",
)
app.DEFINE_float(
  "unroll_factor",
  0,
  "Determine the number of dynamic model unrolls to perform. If "
  "--unroll_strategy=constant, this number of unrolls - each of size sum(layer_timesteps) are performed. "
  "So one unroll adds sum(layer_timesteps) many steps to the network. If "
  "--unroll_strategy=edge_counts, max_edge_count * --unroll_factor timesteps "
  "are performed. (rounded up to the next multiple of sum(layer_timesteps))",
)
# We assume that position_embeddings exist in every dataset.
# the flag now only controls whether they are used or not.
# This could be nice for ablating our model and also debugging with and without.
app.DEFINE_string(
  "position_embeddings",
  "fancy",
  "Whether to use position embeddings as signals for edge order."
  "Options: initial, every, fancy, off"
  "initial takes A (h + pos) at first timestep, every does the same at every timestep"
  "fancy learns another weight matrix B, s.th. propagation is A h + B pos"
  "We expect them to be part of the ds anyway, but you can toggle off their effect.",
)

app.DEFINE_boolean("use_edge_bias", False, "")

# TODO(github.com/ChrisCummins/ProGraML/issues/27): Unused flag.
app.DEFINE_boolean(
  "msg_mean_aggregation",
  True,
  "If true, normalize incoming messages by the number of incoming messages.",
)
app.DEFINE_float(
  "graph_state_dropout", 0.0, "Graph state dropout rate.",
)
app.DEFINE_float(
  "edge_weight_dropout", 0.0, "Edge weight dropout rate.",
)
app.DEFINE_float(
  "output_layer_dropout", 0.0, "Dropout rate on the output layer.",
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
    self.dev = (
      torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    app.Log(1, "Using device %s", self.dev)

    # Instantiate model
    config = GGNNConfig(
      num_classes=self.y_dimensionality,
      has_graph_labels=self.graph_db.graph_y_dimensionality > 0,
    )

    inst2vec_embeddings = node_encoder.GraphEncoder().embeddings_tables[0]
    inst2vec_embeddings = torch.from_numpy(
      np.array(inst2vec_embeddings, dtype=np.float32)
    )
    self.model = GGNNModel(
      config,
      pretrained_embeddings=inst2vec_embeddings,
      test_only=FLAGS.test_only,
    )
    self.model.to(self.dev)

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch of data from an iterator of graphs.

  Returns:
    A single batch of data for feeding into RunBatch().
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
      return batches.Data(graph_ids=[], data=None)

    # Workaround for the fact that graph batcher may read one more graph than
    # actually gets included in the batch.
    if batcher.last_graph:
      graphs = graph_iterator.graphs_read[:-1]
    else:
      graphs = graph_iterator.graphs_read

    return batches.Data(
      graph_ids=[graph.id for graph in graphs],
      data=GgnnBatchData(disjoint_graph=disjoint_graph, graphs=graphs),
    )

  @property
  def message_passing_step_count(self) -> int:
    return self.layer_timesteps.sum()

  @property
  def layer_timesteps(self) -> np.array:
    return np.array([int(x) for x in FLAGS.layer_timesteps])

  # TODO(github.com/ChrisCummins/ProGraML/issues/27): Split this into a separate
  # unroll_strategy.py module.
  def GetUnrollFactor(
    self,
    epoch_type: epoch.Type,
    batch: batches.Data,
    unroll_strategy: str,
    unroll_factor: float,
  ) -> int:
    """Determine the unroll factor from the --unroll_strategy and --unroll_factor
  flags, and the batch log.
  """
    # Determine the unrolling strategy.
    if unroll_strategy == "none" or epoch_type == epoch.Type.TRAIN:
      # Perform no unrolling. The inputs are processed for a single run of
      # message_passing_step_count. This is required during training to
      # propagate gradients.
      return 1
    elif unroll_strategy == "constant":
      # Unroll by a constant number of steps. The total number of steps is
      # (unroll_factor * message_passing_step_count).
      return int(unroll_factor)
    elif unroll_strategy == "data_flow_max_steps":
      max_data_flow_steps = max(
        graph.data_flow_steps for graph in batch.data.disjoint_graphs
      )
      unroll_factor = math.ceil(
        max_data_flow_steps / self.message_passing_step_count
      )
      app.Log(
        2,
        "Determined unroll factor %d from max data flow steps %d",
        unroll_factor,
        max_data_flow_steps,
      )
      return unroll_factor
    elif unroll_strategy == "edge_count":
      max_edge_count = max(graph.edge_count for graph in batch.data.graphs)
      unroll_factor = math.ceil(
        (max_edge_count * unroll_factor) / self.message_passing_step_count
      )
      app.Log(
        2,
        "Determined unroll factor %d from max edge count %d",
        unroll_factor,
        self.message_passing_step_count,
      )
      return unroll_factor
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
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): maybe we can save memory copies in the training loop if we can turn the data into the required types (np.int64 and np.float32) once they come off the network from the database, where smaller i/o size (int32) is more important.
    vocab_ids = torch.from_numpy(
      np.array(disjoint_graph.node_x[:, 0], dtype=np.int64)
    )
    selector_ids = torch.from_numpy(
      np.array(disjoint_graph.node_x[:, 1], dtype=np.int64)
    )
    labels = (
      torch.from_numpy(disjoint_graph.node_y)
      if disjoint_graph.has_node_y
      else torch.from_numpy(disjoint_graph.graph_y)
    )
    edge_lists = [
      torch.from_numpy(np.array(x, dtype=np.int64))
      for x in disjoint_graph.adjacencies
    ]

    # Send data to the GPU.
    with ctx.Profile(5, "Sent data to GPU"):
      vocab_ids.to(self.dev)
      selector_ids.to(self.dev)
      labels.to(self.dev)
      for edge_list in edge_lists:
        edge_list.to(self.dev)

    # TODO(github.com/ChrisCummins/ProGraML/issues/30) still unused
    edge_positions = [
      torch.from_numpy(x) for x in disjoint_graph.edge_positions
    ]

    if disjoint_graph.has_graph_y:
      num_graphs = torch.tensor(disjoint_graph.graph_count, dtype=torch.long)
      graph_nodes_list = torch.from_numpy(
        np.array(disjoint_graph.graph_nodes_list, dtype=np.int64)
      )
      aux_in = torch.from_numpy(
        np.array(disjoint_graph.graph_x, dtype=np.float32)
      )
      num_graphs.to(self.dev)
      graph_nodes_list.to(self.dev)
      aux_in.to(self.dev)
      model_inputs = (
        vocab_ids,
        selector_ids,
        labels,
        edge_lists,
        num_graphs,
        graph_nodes_list,
        aux_in,
      )
    else:
      model_inputs = (vocab_ids, selector_ids, labels, edge_lists)

    # enter correct mode of model
    if epoch_type == epoch.Type.TRAIN and not self.model.training:
      self.model.train()
    elif self.model.training:
      self.model.eval()
      self.model.opt.zero_grad()

    outputs = self.model(*model_inputs)

    logits, accuracy, logits, correct, targets, graph_features = outputs

    loss = self.model.loss((logits, graph_features), targets)

    if epoch_type == epoch.Type.TRAIN:
      loss.backward()
      # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Clip gradients (done). NB, pytorch clips by norm of the gradient of the model, while tf clips by norm of the grad of eeach tensor separately. Therefore we change default from 1.0 to 6.0.
      # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Anyway: Gradients shouldn't really be clipped if not necessary?
      if self.model.config.clip_grad_norm > 0.0:
        nn.utils.clip_grad_norm_(
          self.model.parameters(), self.model.config.clip_grad_norm
        )
      self.model.opt.step()
      self.model.opt.zero_grad()

    loss = loss.item()

    # tg = targets.numpy()
    # tg = np.vstack(((tg + 1) % 2, tg)).T
    # assert np.all(labels.numpy() == tg), f"labels sanity check failed: labels={labels.numpy()},  tg={tg}"

    # TODO(github.com/ChrisCummins/ProGraML/issues/27): Learning rate schedule
    # will change this value.
    learning_rate = self.model.config.lr

    # TODO(github.com/ChrisCummins/ProGraML/issues/27): Set these.
    model_converged = False
    iteration_count = 1

    return batches.Results.Create(
      targets=labels.numpy(),
      predictions=logits.detach().numpy(),
      model_converged=model_converged,
      learning_rate=learning_rate,
      iteration_count=iteration_count,
      loss=loss,
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
