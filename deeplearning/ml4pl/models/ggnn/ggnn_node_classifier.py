"""Train and evaluate a model for node classification."""
import collections
import numpy as np
import tensorflow as tf
import typing

from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.ggnn import ggnn_base as ggnn
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils
from labm8 import app

FLAGS = app.FLAGS

##### Beginning of flag declarations.
#
app.DEFINE_integer(
    'max_instance_count', None,
    'A debugging option. Use this to set the maximum number of instances used '
    'from training/validation/test files. Note this still requires reading '
    'the entirety of the file contents into memory.')

app.DEFINE_string("graph_rnn_cell", "GRU",
                  "The RNN cell type. One of {GRU,CudnnCompatibleGRUCell,RNN}")
ggnn.MODEL_FLAGS.add("graph_rnn_cell")

app.DEFINE_string("graph_rnn_activation", "tanh",
                  "The RNN activation type. One of {tanh,ReLU}")
ggnn.MODEL_FLAGS.add("graph_rnn_activation")

app.DEFINE_boolean("use_propagation_attention", False, "")
ggnn.MODEL_FLAGS.add("use_propagation_attention")

app.DEFINE_boolean("use_edge_bias", False, "")
ggnn.MODEL_FLAGS.add("use_edge_bias")

app.DEFINE_boolean(
    "use_edge_msg_avg_aggregation", True,
    "If true, normalize incoming messages by the number of "
    "incoming messages.")
ggnn.MODEL_FLAGS.add("use_edge_msg_avg_aggregation")

app.DEFINE_float("graph_state_dropout_keep_prob", 1.0,
                 "Graph state dropout keep probability (rate = 1 - keep_prob)")
ggnn.MODEL_FLAGS.add("graph_state_dropout_keep_prob")

app.DEFINE_float("edge_weight_dropout_keep_prob", 1.0,
                 "Edge weight dropout keep probability (rate = 1 - keep_prob)")
ggnn.MODEL_FLAGS.add("edge_weight_dropout_keep_prob")

GGNNWeights = collections.namedtuple(
    "GGNNWeights",
    [
        "edge_weights",
        "edge_biases",
        "edge_type_attention_weights",
        "rnn_cells",
    ],
)

# TODO(cec): Refactor.
residual_connections = {}


class GgnnNodeClassifierModel(ggnn.GgnnBaseModel):
  """GGNN model for learning node classification."""

  def MakeLossAndAccuracyAndPredictionOps(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    layer_timesteps = np.array([int(x) for x in FLAGS.layer_timesteps])
    app.Log(1, "Using layer timesteps: %s for a total step count of %s",
            layer_timesteps, layer_timesteps.sum())

    # Generate per-layer values for edge weights, biases and gated units:
    self.weights = {}  # Used by super-class to place generic things
    self.gnn_weights = GGNNWeights([], [], [], [])
    for layer_index in range(len(self.layer_timesteps)):
      with tf.variable_scope(f"gnn_layer_{layer_index}"):
        edge_weights = tf.reshape(
                tf.Variable(
                    utils.glorot_init([
                        self.stats.edge_type_count * FLAGS.hidden_size,
                        FLAGS.hidden_size
                    ]),
                    name=f"gnn_edge_weights_{layer_index}",
                ), [
                    self.stats.edge_type_count, FLAGS.hidden_size,
                    FLAGS.hidden_size
                ])
        # Add dropout as required.
        if FLAGS.edge_weight_dropout_keep_prob < 1.0:
          edge_weights = tf.nn.dropout(
              edge_weights,
              rate=1 - self.placeholders["edge_weight_dropout_keep_prob"])
        self.gnn_weights.edge_weights.append(edge_weights)

        if FLAGS.use_propagation_attention:
          self.gnn_weights.edge_type_attention_weights.append(
              tf.Variable(
                  np.ones([self.stats.edge_type_count], dtype=np.float32),
                  name=f"edge_type_attention_weights_{layer_index}",
              ))

        if FLAGS.use_edge_bias:
          self.gnn_weights.edge_biases.append(
              tf.Variable(
                  np.zeros([self.stats.edge_type_count, FLAGS.hidden_size],
                           dtype=np.float32),
                  name="gnn_edge_biases_%i" % layer_index,
              ))

        cell = utils.BuildRnnCell(FLAGS.graph_rnn_cell,
                                  FLAGS.graph_rnn_activation,
                                  FLAGS.hidden_size,
                                  name=f"cell_layer_{layer_index}")
        # Apply dropout as required.
        if FLAGS.graph_state_dropout_keep_prob < 1:
          cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
              cell, state_keep_prob=self.placeholders["graph_state_dropout_keep_prob"])
        self.gnn_weights.rnn_cells.append(cell)

    # Initial node states and then one entry per layer
    # (final state of that layer), shape: number of nodes
    # in batch v x D.
    node_states_per_layer = [self.placeholders['node_x']]

    # Number of nodes in batch.
    num_nodes_in_batch = self.placeholders['node_count']

    message_targets = []  # List of tensors of message targets of shape [E]
    message_edge_types = []  # List of tensors of edge type of shape [E]

    # TODO(cec): Investigate this.
    # Each edge type gets a unique edge_type_idx from its own adjacency list.
    # I will have to only carry one adj. list (one edge type, maybe could go to
    # 2 for data and flow) and instead figure out how to carry the emb as
    # additional information, cf. "prep. spec. graphmodel: placeholder def.".
    for edge_type, adjacency_list_for_edge_type in enumerate(
        self.placeholders["adjacency_lists"]):
      edge_targets = adjacency_list_for_edge_type[:, 1]
      message_targets.append(edge_targets)
      message_edge_types.append(
          tf.ones_like(edge_targets, dtype=tf.int32) * edge_type)

    message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
    message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

    for (layer_idx, num_timesteps) in enumerate(self.layer_timesteps):
      with tf.variable_scope("gnn_layer_%i" % layer_idx):
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        # Extract residual messages, if any:
        # TODO(cec): Refactor into separate function.
        layer_residual_connections = residual_connections.get(str(layer_idx))
        if layer_residual_connections is None:
          layer_residual_states = []
        else:
          layer_residual_states = [
              node_states_per_layer[residual_layer_idx]
              for residual_layer_idx in layer_residual_connections
          ]

        if FLAGS.use_propagation_attention:
          message_edge_type_factors = tf.nn.embedding_lookup(
              params=self.gnn_weights.edge_type_attention_weights[layer_idx],
              ids=message_edge_types,
          )  # Shape [M]

        # Record new states for this layer. Initialised to last state, but will
        # be updated below:
        node_states_per_layer.append(node_states_per_layer[-1])
        for step in range(num_timesteps):
          with tf.variable_scope(f"timestep_{step}"):
            # list of tensors of messages of shape [E, D]
            messages = []
            # list of tensors of edge source states of shape [E, D]
            message_source_states = []

            # Collect incoming messages per edge type
            for edge_type, adjacency_list_for_edge_type in enumerate(
                self.placeholders["adjacency_lists"]):
              edge_sources = adjacency_list_for_edge_type[:, 0]
              edge_source_states = tf.nn.embedding_lookup(
                  params=node_states_per_layer[-1],
                  ids=edge_sources)  # Shape [E, D]

              # Message propagation.
              all_messages_for_edge_type = tf.matmul(
                  edge_source_states,
                  self.gnn_weights.edge_weights[layer_idx][edge_type],
              )  # Shape [E, D]
              messages.append(all_messages_for_edge_type)
              message_source_states.append(edge_source_states)

            messages = tf.concat(messages, axis=0)  # Shape [M, D]

            # TODO: not well understood
            if FLAGS.use_propagation_attention:
              message_source_states = tf.concat(message_source_states,
                                                axis=0)  # Shape [M, D]
              message_target_states = tf.nn.embedding_lookup(
                  params=node_states_per_layer[-1],
                  ids=message_targets)  # Shape [M, D]
              message_attention_scores = tf.einsum(
                  "mi,mi->m", message_source_states,
                  message_target_states)  # Shape [M]
              message_attention_scores = (message_attention_scores *
                                          message_edge_type_factors)

              # The following is softmax-ing over the incoming messages per
              # node. As the number of incoming varies, we can't just use
              # tf.softmax. Reimplement with logsumexp trick:
              # Step (1): Obtain shift constant as max of messages going into
              # a node.
              message_attention_score_max_per_target = tf.unsorted_segment_max(
                  data=message_attention_scores,
                  segment_ids=message_targets,
                  num_segments=num_nodes_in_batch,
              )  # Shape [V]
              # Step (2): Distribute max out to the corresponding messages
              # again, and shift scores:
              message_attention_score_max_per_message = tf.gather(
                  params=message_attention_score_max_per_target,
                  indices=message_targets,
              )  # Shape [M]
              message_attention_scores -= (
                  message_attention_score_max_per_message)
              # Step (3): Exp, sum up per target, compute exp(score) / exp(sum)
              # as attention prob:
              message_attention_scores_exped = tf.exp(
                  message_attention_scores)  # Shape [M]
              message_attention_score_sum_per_target = tf.unsorted_segment_sum(
                  data=message_attention_scores_exped,
                  segment_ids=message_targets,
                  num_segments=num_nodes_in_batch,
              )  # Shape [V]
              message_attention_normalisation_sum_per_message = tf.gather(
                  params=message_attention_score_sum_per_target,
                  indices=message_targets,
              )  # Shape [M]
              message_attention = message_attention_scores_exped / (
                  message_attention_normalisation_sum_per_message +
                  utils.SMALL_NUMBER)  # Shape [M]
              # Step (4): Weigh messages using the attention prob:
              messages = messages * tf.expand_dims(message_attention, -1)

            incoming_messages = tf.unsorted_segment_sum(
                data=messages,
                segment_ids=message_targets,
                num_segments=num_nodes_in_batch,
            )  # Shape [V, D]

            if FLAGS.use_edge_bias:
              incoming_messages += tf.matmul(
                  self.placeholders["incoming_edge_counts"],
                  self.gnn_weights.edge_biases[layer_idx],
              )  # Shape [V, D]

            if FLAGS.use_edge_msg_avg_aggregation:
              num_incoming_edges = tf.reduce_sum(
                  self.placeholders["incoming_edge_counts"],
                  keepdims=True,
                  axis=-1,
              )  # Shape [V, 1]
              incoming_messages /= num_incoming_edges + utils.SMALL_NUMBER

            incoming_information = tf.concat(
                layer_residual_states + [incoming_messages],
                axis=-1)  # Shape [V, D*(1 + num of residual connections)]

            # pass updated vertex features into RNN cell, shape [V, D].
            node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](
                incoming_information, node_states_per_layer[-1])[1]

    self.ops["final_node_x"] = node_states_per_layer[-1]

    predictions, regression_gate, regression_transform = (utils.MakeOutputLayer(
        initial_node_state=node_states_per_layer[0],
        final_node_state=self.ops["final_node_x"],
        hidden_size=FLAGS.hidden_size,
        labels_dimensionality=self.stats.node_labels_dimensionality,
        dropout_keep_prob_placeholder=self.
        placeholders["out_layer_dropout_keep_prob"]))
    self.weights['regression_gate'] = regression_gate
    self.weights['regression_transform'] = regression_transform

    targets = tf.argmax(self.placeholders["node_y"],
                        axis=1,
                        output_type=tf.int32)

    accuracies = tf.equal(tf.argmax(predictions, axis=1, output_type=tf.int32),
                          targets)

    accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

    loss = tf.losses.softmax_cross_entropy(self.placeholders["node_y"],
                                           predictions)

    return loss, accuracies, accuracy, predictions

  def MakeMinibatchIterator(
      self, epoch_type: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLog, ggnn.FeedDict]]:
    """Create minibatches by flattening adjacency matrices into a single
    adjacency matrix with multiple disconnected components."""
    for batch in self.batcher.MakeGroupBatchIterator(epoch_type):
      feed_dict = utils.BatchDictToFeedDict(batch, self.placeholders)
      if epoch_type == "train":
        feed_dict.update({
            self.placeholders["graph_state_keep_prob"]:
            (FLAGS.graph_state_dropout_keep_prob),
            self.placeholders["edge_weight_dropout_keep_prob"]:
            (FLAGS.edge_weight_dropout_keep_prob),
            self.placeholders["out_layer_dropout_keep_prob"]:
            (FLAGS.out_layer_dropout_keep_prob)
        })
      else:
        feed_dict.update({
            self.placeholders["graph_state_keep_prob"]: 1.0,
            self.placeholders["edge_weight_dropout_keep_prob"]: 1.0,
            self.placeholders["out_layer_dropout_keep_prob"]: 1.0,
        })
      yield batch['log'], feed_dict


def main():
  """Main entry point."""
  graph_db = FLAGS.graph_db()
  log_db = FLAGS.log_db()
  working_dir = FLAGS.working_dir
  if not working_dir:
    raise app.UsageError("--working_dir is required")

  app.Log(1, 'Using working dir %s', working_dir)

  model = GgnnNodeClassifierModel(graph_db, log_db)
  model.Train()


if __name__ == '__main__':
  app.Run(main)
