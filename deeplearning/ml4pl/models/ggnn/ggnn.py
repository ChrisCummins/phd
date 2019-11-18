"""Train and evaluate a model for node classification."""
import collections
import typing
import warnings

import numpy as np
import tensorflow as tf

from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
from deeplearning.ml4pl.models import base_utils
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.ggnn import ggnn_base as ggnn
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils
from labm8 import app

FLAGS = app.FLAGS

##### Beginning of flag declarations.
#
# Some of these flags define parameters which must be equal when restoring from
# file, such as the hidden layer sizes. Other parameters may change between
# runs of the same model, such as the input data batch size. To accomodate for
# this, a ClassifierBase.GetModelFlagNames() method returns the list of flags
# which must be consistent between runs of the same model.
#
# For the sake of readability, these important model flags are saved into a
# global set classifier_base.MODEL_FLAGS here, so that the declaration of model
# flags is local to the declaration of the flag.
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

app.DEFINE_boolean(
    "use_edge_msg_avg_aggregation", True,
    "If true, normalize incoming messages by the number of "
    "incoming messages.")
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

app.DEFINE_float(
    "intermediate_loss_discount_factor", 0.2,
    "The actual loss is computed as loss + factor * intermediate_loss")

app.DEFINE_integer(
    "auxiliary_inputs_dense_layer_size", 32,
    "Size for MLP that combines graph_x and GGNN output features")
classifier_base.MODEL_FLAGS.add("auxiliary_inputs_dense_layer_size")

app.DEFINE_boolean(
    "use_dsc_loss", False,
    "Whether to use the DSC loss instead of Cross Entropy."
    "DSC loss help with class imbalances. Refer to "
    "https://arxiv.org/pdf/1911.02855.pdf")
app.DEFINE_string(
    "manual_tag", "",
    "An arbitrary tag that can be printed to the leaderboard later.")

###########################
app.DEFINE_boolean('kfold', False,
                   "Set to do automatic kfold validation on devmap.")

app.DEFINE_list('groups', [str(x) for x in range(10)],
                'The test groups to use.')

###########################

GGNNWeights = collections.namedtuple(
    "GGNNWeights",
    [
        "edge_weights",
        "edge_biases",
        "edge_type_attention_weights",
        "rnn_cells",
    ],
)


class GgnnClassifier(ggnn.GgnnBaseModel):
  """GGNN model for node-level or graph-level classification."""

  def MakeLossAndAccuracyAndPredictionOps(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    layer_timesteps = np.array([int(x) for x in FLAGS.layer_timesteps])
    app.Log(
        1, "Using layer timesteps: %s for a total of %s message passing "
        "steps", layer_timesteps, self.message_passing_step_count)

    # Generate per-layer values for edge weights, biases and gated units:
    self.weights = {}  # Used by super-class to place generic things
    self.gnn_weights = GGNNWeights([], [], [], [])

    for layer_index in range(len(self.layer_timesteps)):
      with tf.compat.v1.variable_scope(f"gnn_layer_{layer_index}"):
        # position propagation matrices are treated like another edge type
        if FLAGS.position_embeddings == 'fancy':
          type_count_with_fancy = 1 + self.stats.edge_type_count
        else:
          type_count_with_fancy = self.stats.edge_type_count

        edge_weights = tf.reshape(
            tf.Variable(
                utils.glorot_init([
                    type_count_with_fancy * FLAGS.hidden_size, FLAGS.hidden_size
                ]),
                name=f"gnn_edge_weights_{layer_index}",
            ), [type_count_with_fancy, FLAGS.hidden_size, FLAGS.hidden_size])

        # Add dropout as required.
        if FLAGS.edge_weight_dropout_keep_prob < 1.0:
          edge_weights = tf.nn.dropout(
              edge_weights,
              rate=1 - self.placeholders["edge_weight_dropout_keep_prob"])
        self.gnn_weights.edge_weights.append(edge_weights)

        if FLAGS.use_propagation_attention:
          self.gnn_weights.edge_type_attention_weights.append(
              tf.Variable(
                  np.ones([type_count_with_fancy], dtype=np.float32),
                  name=f"edge_type_attention_weights_{layer_index}",
              ))

        if FLAGS.use_edge_bias:
          self.gnn_weights.edge_biases.append(
              tf.Variable(
                  np.zeros([type_count_with_fancy, FLAGS.hidden_size],
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
              cell,
              state_keep_prob=self.placeholders["graph_state_dropout_keep_prob"]
          )
        self.gnn_weights.rnn_cells.append(cell)
      # end of variable scope f"gnn_layer_{layer_index}"

    with tf.compat.v1.variable_scope("embeddings"):
      # maybe generate table with position embs up to pos 512.
      if FLAGS.position_embeddings != 'off':
        self.position_embeddings = self._GetPositionEmbeddingsAsTensorflowVariable(
        )

      # Lookup each node embedding table and concatenate the result.
      embeddings = self._GetEmbeddingsAsTensorflowVariables()
      for i in range(len(embeddings)):
        self.weights[f'node_embeddings_{i}'] = embeddings[i]

      encoded_node_x = tf.compat.v1.concat([
          tf.nn.embedding_lookup(self.weights[f'node_embeddings_{i}'],
                                 ids=self.placeholders['node_x'][:, i])
          for i in range(len(embeddings))
      ],
                                           axis=1,
                                           name='embeddings_concat')

    ###########################################################################
    ###  GGNN UNROLLING START

    # Initial node states and then one entry per layer
    # (final state of that layer), shape: number of nodes
    # in batch v x D.
    node_states_per_layer = [encoded_node_x]

    # Number of nodes in batch.
    num_nodes_in_batch = self.placeholders['node_count']

    message_targets = []  # List of tensors of message targets of shape [E]
    message_edge_types = []  # List of tensors of edge type of shape [E]
    for edge_type, adjacency_list in enumerate(
        self.placeholders['adjacency_lists']):
      edge_targets = adjacency_list[:, 1]
      message_targets.append(edge_targets)
      message_edge_types.append(
          tf.ones_like(edge_targets, dtype=tf.int32) * edge_type)

    message_targets = tf.concat(message_targets, axis=0,
                                name='message_targets')  # Shape [M]
    message_edge_types = tf.concat(message_edge_types,
                                   axis=0,
                                   name='message_edge_types')  # Shape [M]

    for (layer_idx, num_timesteps) in enumerate(self.layer_timesteps):
      with tf.compat.v1.variable_scope(f"gnn_layer_{layer_idx}"):
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        if FLAGS.use_propagation_attention:
          message_edge_type_factors = tf.nn.embedding_lookup(
              params=self.gnn_weights.edge_type_attention_weights[layer_idx],
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
            for edge_type, (adjacency_list, edge_positions) in enumerate(
                zip(self.placeholders["adjacency_lists"],
                    self.placeholders['edge_positions'])):
              edge_sources = adjacency_list[:, 0]

              edge_source_states = tf.nn.embedding_lookup(
                  params=node_states_per_layer[-1],
                  ids=edge_sources)  # Shape [E, D]

              if FLAGS.position_embeddings != 'off':
                edge_pos_embedding = tf.nn.embedding_lookup(
                    self.position_embeddings,
                    ids=edge_positions)  # shape [E, D]

              # one among: {initial, every, fancy, off}
              # maybe add position to edge_source_states here
              if FLAGS.position_embeddings == 'every' or \
                (step and FLAGS.position_embeddings == 'initial'):
                edge_source_states = tf.add(
                    edge_source_states,
                    edge_pos_embedding,
                    name='edge_source_states_with_position')

              # Message propagation.
              # Term: A * h
              all_messages_for_edge_type = tf.matmul(
                  edge_source_states,
                  self.gnn_weights.edge_weights[layer_idx][edge_type],
              )  # Shape [E, D]

              # maybe add term B * pos
              if FLAGS.position_embeddings == 'fancy':
                all_messages_for_edge_type += tf.matmul(
                    edge_pos_embedding,
                    # last edge_type corresponds to fancy_position_weights B
                    self.gnn_weights.edge_weights[layer_idx][-1],
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

            # Shape [V, D]
            incoming_information = tf.concat([incoming_messages], axis=-1)

            # pass updated vertex features into RNN cell, shape [V, D].
            node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](
                incoming_information, node_states_per_layer[-1])[1]

    self.ops["final_node_x"] = node_states_per_layer[-1]

    ###  GGNN UNROLLING END
    ###########################################################################
    ###  GGNN READOUT START

    if FLAGS.output_layer_dropout_keep_prob < 1:
      out_layer_dropout = self.placeholders["output_layer_dropout_keep_prob"]
    else:
      out_layer_dropout = None

    labels_dimensionality = (self.stats.node_labels_dimensionality or
                             self.stats.graph_labels_dimensionality)
    predictions, regression_gate, regression_transform = utils.MakeOutputLayer(
        initial_node_state=node_states_per_layer[0],
        final_node_state=self.ops["final_node_x"],
        hidden_size=FLAGS.hidden_size,
        labels_dimensionality=labels_dimensionality,
        dropout_keep_prob_placeholder=out_layer_dropout)
    self.weights['regression_gate'] = regression_gate
    self.weights['regression_transform'] = regression_transform

    if self.stats.graph_features_dimensionality:
      # Sum node representations across graph (per graph).
      computed_graph_only_values = tf.unsorted_segment_sum(
          predictions,
          segment_ids=self.placeholders["graph_nodes_list"],
          num_segments=self.placeholders["graph_count"],
          name='computed_graph_only_values',
      )  # [g, c]

      # Add global features to the graph readout.
      x = tf.concat([
          computed_graph_only_values,
          tf.cast(self.placeholders["graph_x"], tf.float32)
      ],
                    axis=-1)
      x = tf.layers.batch_normalization(
          x, training=self.placeholders['is_training'])
      x = tf.layers.dense(x,
                          FLAGS.auxiliary_inputs_dense_layer_size,
                          activation=tf.nn.relu)
      x = tf.layers.dropout(
          x,
          rate=1 - self.placeholders["output_layer_dropout_keep_prob"],
          training=self.placeholders['is_training'])
      predictions = tf.layers.dense(x, 2)

    if self.stats.graph_labels_dimensionality:
      targets = tf.argmax(self.placeholders["graph_y"],
                          axis=1,
                          output_type=tf.int32,
                          name="targets")
    elif self.stats.node_labels_dimensionality:
      targets = tf.argmax(self.placeholders["node_y"],
                          axis=1,
                          output_type=tf.int32)
    else:
      raise ValueError("No graph labels and no node labels!")

    argmaxed_predictions = tf.argmax(predictions, axis=1, output_type=tf.int32)
    accuracies = tf.equal(argmaxed_predictions, targets)

    accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

    if self.stats.graph_labels_dimensionality:
      graph_only_loss = tf.losses.softmax_cross_entropy(
          self.placeholders["graph_y"], computed_graph_only_values)
      _loss = tf.losses.softmax_cross_entropy(self.placeholders["graph_y"],
                                              predictions)
      loss = _loss + FLAGS.intermediate_loss_discount_factor * graph_only_loss
    elif self.stats.node_labels_dimensionality:
      if FLAGS.use_dsc_loss:
        # self.placeholders['node_y'] have shape (num_nodes_in_batch, 2)
        p1 = tf.nn.softmax(predictions[:, 0])
        y1 = tf.cast(self.placeholders['node_y'][:, 0], tf.float32)

        # we fix class 2 bc here 0 is the dominant mode!
        p2 = 1.0 - tf.nn.softmax(predictions[:, 1])
        y2 = 1.0 - tf.cast(self.placeholders['node_y'][:, 1], tf.float32)

        loss = (self.make_dsc_loss(p1, y1) + self.make_dsc_loss(p2, y2)) / 2.0
      else:
        loss = tf.losses.softmax_cross_entropy(self.placeholders["node_y"],
                                               predictions)
        #loss = 0.0

    else:
      raise ValueError("No graph labels and no node labels!")

    return loss, accuracies, accuracy, predictions

  def make_dsc_loss(self, p1: tf.Tensor, y1: tf.Tensor):
    normalization = tf.cast(self.placeholders['node_count'], tf.float32)
    numerator = (1. - p1) * p1 * y1
    denominator = ((1.0 - p1) * p1 + y1) * normalization
    neg_loss = tf.reduce_sum(numerator / (denominator + utils.SMALL_NUMBER))
    return 1.0 - neg_loss

  def MakeMinibatchIterator(
      self, epoch_type: str, groups: typing.List[str], print_context: typing.Any = None
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, ggnn.FeedDict]]:
    """Create mini-batches by flattening adjacency matrices into a single
    adjacency matrix with multiple disconnected components."""
    options = graph_batcher.GraphBatchOptions(
        max_nodes=FLAGS.batch_size,
        groups=groups,
        data_flow_max_steps_required=(None if epoch_type == 'test' else
                                      self.message_passing_step_count))
    max_instance_count = (
        FLAGS.max_train_per_epoch if epoch_type == 'train' else
        FLAGS.max_val_per_epoch if epoch_type == 'val' else None)
    for batch in self.batcher.MakeGraphBatchIterator(options,
                                                     max_instance_count, print_context=print_context):
      feed_dict = utils.BatchDictToFeedDict(batch, self.placeholders)

      if epoch_type == "train":
        feed_dict.update({
            self.placeholders["graph_state_dropout_keep_prob"]:
            FLAGS.graph_state_dropout_keep_prob,
            self.placeholders["edge_weight_dropout_keep_prob"]:
            FLAGS.edge_weight_dropout_keep_prob,
            self.placeholders["output_layer_dropout_keep_prob"]:
            FLAGS.output_layer_dropout_keep_prob,
            self.placeholders["is_training"]:
            True,
            self.placeholders['learning_rate_multiple']:
            base_utils.WarmUpAndFinetuneLearningRateSchedule(
                self.epoch_num, FLAGS.num_epochs)
            if FLAGS.use_lr_schedule else 1.0
        })
      else:
        feed_dict.update({
            self.placeholders["graph_state_dropout_keep_prob"]: 1.0,
            self.placeholders["edge_weight_dropout_keep_prob"]: 1.0,
            self.placeholders["output_layer_dropout_keep_prob"]: 1.0,
            self.placeholders["is_training"]: False,
        })
      yield batch.log, feed_dict

  def MakeModularGraphOps(self):
    if not (self.weights['regression_gate'] and
            self.weights['regression_transform']):
      raise TypeError("MakeModularGraphOps() call before "
                      "MakeLossAndAccuracyAndPredictionOps()")

    predictions = utils.MakeModularOutputLayer(
        self.placeholders['node_x'],
        self.placeholders['raw_node_output_features'],
        self.weights['regression_gate'], self.weights['regression_transform'])

    targets = tf.argmax(self.placeholders["node_y"],
                        axis=1,
                        output_type=tf.int32)

    accuracies = tf.equal(tf.argmax(predictions, axis=1, output_type=tf.int32),
                          targets)

    accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

    loss = tf.losses.softmax_cross_entropy(self.placeholders["node_y"],
                                           predictions)

    return loss, accuracies, accuracy, predictions


def RunKFoldOrDie():
  for test_group in FLAGS.groups:
    app.Log(1, 'Testing group %s on database %s', test_group, FLAGS.graph_db)

    test_group_as_num = int(test_group)
    assert 10 > test_group_as_num >= 0
    val_group = str((test_group_as_num + 1) % 10)

    FLAGS.test_group = test_group
    FLAGS.val_group = val_group
    classifier_base.Run(GgnnClassifier)


def main():
  """Main entry point."""
  # TODO(cec): Only filter https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.UndefinedMetricWarning.html
  warnings.filterwarnings("ignore")

  if not FLAGS.log_db:
    app.FatalWithoutStackTrace("--log_db must be set")
  if not FLAGS.working_dir:
    app.FatalWithoutStackTrace("--working_dir must be set")
  if FLAGS.position_embeddings not in ['initial', 'every', 'fancy', 'off']:
    app.FatalWithoutStackTrace(
        "--position_embeddings has to be one of <initial, every, fancy, off>")

  app.Log = base_utils.AppLogWrapper()
  if not FLAGS.kfold:
    classifier_base.Run(GgnnClassifier)
  else:
    app.Log(1, "Running kfold on test groups %s", FLAGS.groups)
    RunKFoldOrDie()


if __name__ == '__main__':
  app.Run(main)
