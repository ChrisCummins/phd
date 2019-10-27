"""Train and evaluate a model for graph classification."""
import collections
import typing

import numpy as np
import tensorflow as tf
from labm8 import app

from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.ggnn import ggnn_base as ggnn
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils

FLAGS = app.FLAGS

##### Beginning of flag declarations.
#
app.DEFINE_string("graph_rnn_cell", "GRU",
                  "The RNN cell type. One of {GRU,CudnnCompatibleGRUCell,RNN}")
classifier_base.MODEL_FLAGS.add("graph_rnn_cell")

app.DEFINE_string("graph_rnn_activation", "tanh",
                  "The RNN activation type. One of {tanh,ReLU}")
classifier_base.MODEL_FLAGS.add("graph_rnn_activation")

app.DEFINE_boolean("use_propagation_attention", False, "")
classifier_base.MODEL_FLAGS.add("use_propagation_attention")

app.DEFINE_boolean("use_edge_bias", False, "")
classifier_base.MODEL_FLAGS.add("use_edge_bias")

app.DEFINE_boolean("use_edge_msg_avg_aggregation", True, "")
classifier_base.MODEL_FLAGS.add("use_edge_msg_avg_aggregation")

app.DEFINE_float("graph_state_dropout_keep_prob", 1.0,
                 "Graph state dropout keep probability (rate = 1 - keep_prob)")
classifier_base.MODEL_FLAGS.add("graph_state_dropout_keep_prob")

app.DEFINE_float("edge_weight_dropout_keep_prob", 1.0,
                 "Edge weight dropout keep probability (rate = 1 - keep_prob)")
classifier_base.MODEL_FLAGS.add("edge_weight_dropout_keep_prob")

app.DEFINE_float(
    "output_layer_dropout_keep_prob", 1.0,
    "Dropout keep probability on the output layer. In range 0 < x <= 1.")
classifier_base.MODEL_FLAGS.add("output_layer_dropout_keep_prob")

app.DEFINE_boolean('ignore_node_features', True, '???')

#
##### End of flag declarations.

GGNNWeights = collections.namedtuple(
    "GGNNWeights",
    [
        "edge_weights",
        "edge_weights_for_embs",
        "edge_biases",
        "edge_biases_for_embs",
        "edge_type_attention_weights",
        "rnn_cells",
    ],
)

# TODO(cec): Refactor.
residual_connections = {}


class GgnnGraphClassifierModel(classifier_base.ClassifierBase):

  def MakeLossAndAccuracyAndPredictionOps(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    layer_timesteps = np.array([int(x) for x in FLAGS.layer_timesteps])
    app.Log(1, "Using layer timesteps: %s for a total step count of %s",
            layer_timesteps, self.message_passing_step_count)

    # Generate per-layer values for edge weights, biases and gated units:
    self.gnn_weights = GGNNWeights([], [], [], [], [], [])
    for layer_index in range(len(layer_timesteps)):
      with tf.compat.v1.variable_scope(f"gnn_layer_{layer_index}"):
        edge_weights = tf.nn.dropout(
            tf.reshape(
                tf.Variable(
                    utils.glorot_init([
                        self.stats.edge_type_count * FLAGS.hidden_size,
                        FLAGS.hidden_size
                    ]),
                    name=f"gnn_edge_weights_{layer_index}",
                ), [
                    self.stats.edge_type_count, FLAGS.hidden_size,
                    FLAGS.hidden_size
                ]),
            rate=1 - self.placeholders["edge_weight_dropout_keep_prob"])
        self.gnn_weights.edge_weights.append(edge_weights)

        # analogous to how edge_weights (for mult. with neighbor states) looked
        # like. this is where we designed the update func. to be:
        # U_m = A*s_n + B*e_(n,m) for all neighbors n of m.
        edge_weights_for_emb = tf.nn.dropout(
            tf.reshape(
                tf.Variable(
                    utils.glorot_init([
                        self.stats.edge_type_count * FLAGS.hidden_size,
                        FLAGS.hidden_size
                    ]),
                    name=f"gnn_edge_weights_for_emb_{layer_index}",
                ), [
                    self.stats.edge_type_count, FLAGS.hidden_size,
                    FLAGS.hidden_size
                ]),
            rate=1 - self.placeholders["edge_weight_dropout_keep_prob"])
        self.gnn_weights.edge_weights_for_embs.append(edge_weights_for_emb)

        if FLAGS.use_propagation_attention:
          self.gnn_weights.edge_type_attention_weights.append(
              tf.Variable(
                  np.ones([self.stats.edge_type_count], dtype=np.float32),
                  name="edge_type_attention_weights_%i" % layer_index,
              ))

        if FLAGS.use_edge_bias:
          self.gnn_weights.edge_biases.append(
              tf.Variable(
                  np.zeros([self.stats.edge_type_count, FLAGS.hidden_size],
                           dtype=np.float32),
                  name="gnn_edge_biases_%i" % layer_index,
              ))
          self.gnn_weights.edge_biases_for_embs.append(
              tf.Variable(
                  np.zeros([self.stats.edge_type_count, FLAGS.hidden_size],
                           dtype=np.float32),
                  name="gnn_edge_biases_%i" % layer_index,
              ))

        cell = utils.BuildRnnCell(FLAGS.graph_rnn_cell,
                                  FLAGS.graph_rnn_activation,
                                  FLAGS.hidden_size,
                                  name=f"cell_layer_{layer_index}")
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            cell, state_keep_prob=self.placeholders["graph_state_keep_prob"])
        self.gnn_weights.rnn_cells.append(cell)

    # TODO(cec): Next chunk of code from compute_final_node_x() ...
    # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
    node_states_per_layer = []

    # number of nodes in batch
    num_nodes = self.placeholders['node_count']

    # we drop the initial states placeholder in the first layer as they are all zero.
    node_states_per_layer.append(
        tf.zeros(
            [num_nodes, FLAGS.hidden_size],
            dtype=tf.float32,
        ))

    message_targets = []  # list of tensors of message targets of shape [E]
    message_edge_types = []  # list of tensors of edge type of shape [E]

    # TODO(cec): Investigate this.
    # Each edge type gets a unique edge_type_idx from its own adjacency list.
    # I will have to only carry one adj. list (one edge type, maybe could go to
    # 2 for data and flow) and instead figure out how to carry the emb as
    # additional information, cf. "prep. spec. graphmodel: placeholder def.".
    for edge_type_idx, adjacency_list in enumerate(
        self.placeholders["adjacency_lists"]):
      edge_targets = adjacency_list[:, 1]
      message_targets.append(edge_targets)
      message_edge_types.append(
          tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)

    message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
    message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

    for (layer_index, num_timesteps) in enumerate(layer_timesteps):
      with tf.compat.v1.variable_scope("gnn_layer_%i" % layer_index):
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        # Extract residual messages, if any:
        layer_residual_connections = residual_connections.get(str(layer_index))
        if layer_residual_connections is None:
          layer_residual_states = []
        else:
          layer_residual_states = [
              node_states_per_layer[residual_layer_index]
              for residual_layer_index in layer_residual_connections
          ]

        if FLAGS.use_propagation_attention:
          message_edge_type_factors = tf.nn.embedding_lookup(
              params=self.gnn_weights.edge_type_attention_weights[layer_index],
              ids=message_edge_types,
          )  # Shape [M]

        # Record new states for this layer. Initialised to last state, but will
        # be updated below:
        node_states_per_layer.append(node_states_per_layer[-1])
        for step in range(num_timesteps):
          with tf.compat.v1.variable_scope(f"timestep_{step}"):
            # list of tensors of messages of shape [E, D]
            messages = []
            # list of tensors of edge source states of shape [E, D]
            message_source_states = []

            # Collect incoming messages per edge type
            for edge_type_idx, (adjacency_lists, edge_embeddings) in enumerate(
                zip(self.placeholders["adjacency_lists"],
                    self.placeholders["edge_x"])):
              edge_sources = adjacency_lists[:, 0]
              # TODO(cec): Hard-coded conversion from edge features to embedding
              # indices.
              edge_emb_idxs = edge_embeddings[:, 0]
              edge_embs = tf.nn.embedding_lookup(
                  params=self.weights["embedding_table"], ids=edge_emb_idxs)
              edge_source_states = tf.nn.embedding_lookup(
                  params=node_states_per_layer[-1],
                  ids=edge_sources)  # Shape [E, D]

              # Message propagation.
              all_messages_for_edge_type = tf.matmul(
                  edge_source_states,
                  self.gnn_weights.edge_weights[layer_index][edge_type_idx],
              ) + tf.matmul(
                  edge_embs,
                  self.gnn_weights.edge_weights[layer_index][edge_type_idx],
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
                  num_segments=num_nodes,
              )  # Shape [V]
              # Step (2): Distribute max out to the corresponding messages again, and shift scores:
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
                  num_segments=num_nodes,
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
                num_segments=num_nodes,
            )  # Shape [V, D]

            if FLAGS.use_edge_bias:
              incoming_messages += tf.matmul(
                  self.placeholders["incoming_edge_counts"],
                  self.gnn_weights.edge_biases[layer_index],
              )  # Shape [V, D]

            if FLAGS.use_edge_msg_avg_aggregation:
              num_incoming_edges = tf.reduce_sum(
                  self.placeholders["incoming_edge_counts"],
                  keep_dims=True,
                  axis=-1,
              )  # Shape [V, 1]
              incoming_messages /= num_incoming_edges + utils.SMALL_NUMBER

            incoming_information = tf.concat(
                layer_residual_states + [incoming_messages],
                axis=-1)  # Shape [V, D*(1 + num of residual connections)]

            # pass updated vertex features into RNN cell
            node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_index](
                incoming_information,
                node_states_per_layer[-1])[1]  # Shape [V, D]

    self.ops["final_node_x"] = node_states_per_layer[-1]

    if FLAGS.output_layer_dropout_keep_prob < 1:
      out_layer_dropout = self.placeholders["output_layer_dropout_keep_prob"]
    else:
      out_layer_dropout = None

    predictions, regression_gate, regression_transform = utils.MakeOutputLayer(
        initial_node_state=tf.zeros(
            [self.placeholders['node_count'], FLAGS.hidden_size]),
        final_node_state=self.ops["final_node_x"],
        hidden_size=FLAGS.hidden_size,
        labels_dimensionality=self.stats.graph_labels_dimensionality,
        dropout_keep_prob_placeholder=out_layer_dropout)
    self.weights['regression_gate'] = regression_gate
    self.weights['regression_transform'] = regression_transform

    # Sum node representations across graph.
    computed_values = tf.unsorted_segment_sum(
        predictions,
        segment_ids=self.placeholders["graph_nodes_list"],
        num_segments=self.placeholders["graph_count"],
        name='computed_values',
    )  # [g, c]

    predictions = tf.argmax(computed_values,
                            axis=1,
                            output_type=tf.int32,
                            name="predictions")

    targets = tf.argmax(self.placeholders["graph_y"],
                        axis=1,
                        output_type=tf.int32,
                        name="targets")

    accuracies = tf.equal(predictions, targets)

    accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

    loss = tf.losses.softmax_cross_entropy(self.placeholders["graph_y"],
                                           computed_values)

    return loss, accuracies, accuracy, computed_values

  def MakeMinibatchIterator(
      self, epoch_type: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLog, ggnn.FeedDict]]:
    """Create minibatches by flattening adjacency matrices into a single
    adjacency matrix with multiple disconnected components."""
    for batch in self.batcher.MakeGroupBatchIterator(epoch_type):
      # Pad node feature vector of size <= hidden_size up to hidden_size so
      # that the size matches embedding dimensionality.
      if 'node_x' in batch:
        batch['node_x'] = np.pad(
            batch["node_x"],
            ((0, 0),
             (0, FLAGS.hidden_size - self.stats.node_features_dimensionality)),
            "constant",
        )

      feed_dict = utils.BatchDictToFeedDict(batch, self.placeholders)

      if epoch_type == "train":
        feed_dict.update({
            self.placeholders["graph_state_keep_prob"]:
            (FLAGS.graph_state_dropout_keep_prob),
            self.placeholders["edge_weight_dropout_keep_prob"]:
            (FLAGS.edge_weight_dropout_keep_prob),
            self.placeholders["output_layer_dropout_keep_prob"]:
            (FLAGS.output_layer_dropout_keep_prob)
        })
      else:
        feed_dict.update({
            self.placeholders["graph_state_keep_prob"]:
            1.0,
            self.placeholders["edge_weight_dropout_keep_prob"]:
            1.0,
            self.placeholders["output_layer_dropout_keep_prob"]:
            1.0,
        })
      yield batch['log'], feed_dict


def main():
  graph_db = FLAGS.graph_db()
  log_db = FLAGS.log_db()
  working_dir = FLAGS.working_dir
  if not working_dir:
    raise app.UsageError("--working_dir is required")

  app.Log(1, 'Using working dir %s', working_dir)

  model = GgnnGraphClassifierModel(graph_db, log_db)
  model.Train()


if __name__ == '__main__':
  app.Run(main)
