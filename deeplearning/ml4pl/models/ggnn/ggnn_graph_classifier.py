"""Train and evaluate a model for graph classification."""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import typing

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as graph_readers
from deeplearning.ml4pl.graphs import graph_database_stats as graph_stats
from deeplearning.ml4pl.graphs.labelled.graph_dict import \
  graph_dict as graph_dicts
from deeplearning.ml4pl.models.ggnn import ggnn_base as ggnn
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils
from labm8 import app
from labm8 import humanize
from labm8 import prof


FLAGS = app.FLAGS

##### Beginning of flag declarations.
#
app.DEFINE_integer("batch_size", 8000,
                   "The maximum number of nodes to include in a graph batch.")

app.DEFINE_database(
    'graph_db',
    graph_database.Database,
    None,
    'The database to read graph data from.',
    must_exist=True)

app.DEFINE_list(
    'layer_timesteps', ['2', '2', '2'],
    'The number of timesteps to propagate for each layer')
ggnn.MODEL_FLAGS.add("layer_timesteps")

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

app.DEFINE_boolean("use_edge_msg_avg_aggregation", True, "")
ggnn.MODEL_FLAGS.add("use_edge_msg_avg_aggregation")

app.DEFINE_float("graph_state_dropout_keep_prob", 1.0,
                 "Graph state dropout keep probability (rate = 1 - keep_prob)")
ggnn.MODEL_FLAGS.add("graph_state_dropout_keep_prob")

app.DEFINE_float("edge_weight_dropout_keep_prob", 1.0,
                 "Edge weight dropout keep probability (rate = 1 - keep_prob)")
ggnn.MODEL_FLAGS.add("edge_weight_dropout_keep_prob")

app.DEFINE_boolean('ignore_node_features', True, '???')

#
##### End of flag declarations.

GGNNWeights = namedtuple(
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


residual_connections = {}

class GgnnGraphClassifierModel(ggnn.GgnnBaseModel):

  def __init__(self, db: graph_database.Database):
    self.db = db
    self.stats = graph_stats.GraphDatabaseStats(self.db)
    super(GgnnGraphClassifierModel, self).__init__()
    app.Log(1, "%s", self.stats)

  def MakeLossAndAccuracyAndPredictionOps(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    layer_timesteps = np.array([int(x) for x in FLAGS.layer_timesteps])
    app.Log(1, "Using layer timesteps: %s for a total step count of %s",
            layer_timesteps, np.prod(layer_timesteps))

    self.placeholders["target_values"] = tf.placeholder(
        tf.int32, [None], name="target_values")

    self.placeholders["num_of_nodes_in_batch"] = tf.placeholder(
        tf.int32, [], name="num_of_nodes_in_batch")

    self.placeholders["adjacency_lists"] = [
        tf.placeholder(tf.int32, [None, 2], name=f"adjacency_e{i}")
        for i in range(self.stats.edge_type_count)
    ]

    self.placeholders["edge_x"] = [
      tf.placeholder(tf.int32,
                     [None, self.stats.edge_features_dimensionality],
                     name=f"edge_x_e{i}")
      for i in range(self.stats.edge_type_count)
    ]

    self.placeholders["num_incoming_edges_per_type"] = tf.placeholder(
        tf.float32, [None, self.stats.edge_type_count],
        name="num_incoming_edges_per_type")

    self.placeholders["graph_nodes_list"] = tf.placeholder(
        tf.int32, [None], name="graph_nodes_list")

    self.placeholders["graph_state_keep_prob"] = tf.placeholder(
        tf.float32, None, name="graph_state_keep_prob")

    self.placeholders["edge_weight_dropout_keep_prob"] = tf.placeholder(
        tf.float32, None, name="edge_weight_dropout_keep_prob")

    # Generate per-layer values for edge weights, biases and gated units:
    self.gnn_weights = GGNNWeights([], [], [], [], [], [])
    for layer_index in range(len(layer_timesteps)):
      with tf.variable_scope(f"gnn_layer_{layer_index}"):
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

        # analogous to how edge_weights (for mult. with neighbor states) looked like.
        # this is where we designed the update func. to be:
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
        self.gnn_weights.edge_weights_for_embs.append(edge_weights)

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

        cell = utils.BuildRnnCell(FLAGS.graph_rnn_cell, FLAGS.graph_rnn_activation,
                                  FLAGS.hidden_size, name=f"cell_layer_{layer_index}")
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            cell, state_keep_prob=self.placeholders["graph_state_keep_prob"])
        self.gnn_weights.rnn_cells.append(cell)

    # TODO(cec): Next chunk of code from compute_final_node_representations() ...
    # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
    node_states_per_layer = []

    # number of nodes in batch
    num_nodes = self.placeholders['num_of_nodes_in_batch']

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
      with tf.variable_scope("gnn_layer_%i" % layer_index):
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
          with tf.variable_scope(f"timestep_{step}"):
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
                  params=self.weights["embedding_table"],
                  ids=edge_emb_idxs)
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
              message_source_states = tf.concat(
                  message_source_states, axis=0)  # Shape [M, D]
              message_target_states = tf.nn.embedding_lookup(
                  params=node_states_per_layer[-1],
                  ids=message_targets)  # Shape [M, D]
              message_attention_scores = tf.einsum(
                  "mi,mi->m", message_source_states,
                  message_target_states)  # Shape [M]
              message_attention_scores = (
                  message_attention_scores * message_edge_type_factors)

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
                  message_attention_normalisation_sum_per_message + utils.SMALL_NUMBER
              )  # Shape [M]
              # Step (4): Weigh messages using the attention prob:
              messages = messages * tf.expand_dims(message_attention, -1)

            incoming_messages = tf.unsorted_segment_sum(
                data=messages,
                segment_ids=message_targets,
                num_segments=num_nodes,
            )  # Shape [V, D]

            if FLAGS.use_edge_bias:
              incoming_messages += tf.matmul(
                  self.placeholders["num_incoming_edges_per_type"],
                  self.gnn_weights.edge_biases[layer_index],
              )  # Shape [V, D]

            if FLAGS.use_edge_msg_avg_aggregation:
              num_incoming_edges = tf.reduce_sum(
                  self.placeholders["num_incoming_edges_per_type"],
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

    self.ops["final_node_representations"] = node_states_per_layer[-1]

    with tf.variable_scope("output_layer"):
      with tf.variable_scope("regression_gate"):
        self.weights["regression_gate"] = utils.MLP(
            # Concatenation of initial and final node states
            in_size=2 * FLAGS.hidden_size,
            out_size=104, # TODO(cec): Un-hard-code to: self.stats.graph_labels_dimensionality,
            hid_sizes=[],
            dropout_keep_prob=self.placeholders["out_layer_dropout_keep_prob"],
        )
        with tf.variable_scope("regression"):
          self.weights["regression_transform"] = utils.MLP(
              in_size=FLAGS.hidden_size,
              out_size=104, # TODO(cec): Un-hard-code to: self.stats.graph_labels_dimensionality,
              hid_sizes=[],
              dropout_keep_prob=self.placeholders["out_layer_dropout_keep_prob"],
          )

        # this is all Eq. 7 in the GGNN paper here... (i.e. Eq. 4 in NMP for QC)
        computed_values = self.GatedRegression(
            self.ops["final_node_representations"],
            self.weights["regression_gate"],
            self.weights["regression_transform"],
            name="computed_values"
        )

        predictions = tf.argmax(computed_values, axis=1, output_type=tf.int32,
                                name="predictions")

        targets = self.placeholders["target_values"]

        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, targets), tf.float32))

        # cross-entropy loss
        loss = tf.losses.sparse_softmax_cross_entropy(targets, computed_values)

        return loss, accuracy, predictions


  def GatedRegression(self, last_h, regression_gate, regression_transform,
                      name: typing.Optional[str] = None):
    # last_h: [v x h]

    if FLAGS.ignore_node_features:
      num_nodes = self.placeholders['num_of_nodes_in_batch']
      initial_node_rep = tf.zeros([num_nodes, FLAGS.hidden_size])
    else:
      initial_node_rep = self.placeholders["initial_node_representation"]

    gate_input = tf.concat([last_h, initial_node_rep], axis=-1)  # [v x 2h]
    gated_outputs = (tf.nn.sigmoid(regression_gate(gate_input)) *
                     regression_transform(last_h))  # [v x 1]

    # Sum up all nodes per-graph
    return tf.unsorted_segment_sum(
        data=gated_outputs,
        segment_ids=self.placeholders["graph_nodes_list"],
        num_segments=self.placeholders["num_graphs"],
        name=name,
    )  # [g, c]


  def MakeMinibatchIterator(
      self,
      epoch_type: str) -> typing.Iterable[typing.Tuple[int, ggnn.FeedDict]]:
    """Create minibatches by flattening adjacency matrices into a single
    adjacency matrix with multiple disconnected components."""
    # Graphs are keyed into {train,val,test} groups.
    filters = [lambda: graph_database.GraphMeta.group == epoch_type]

    graph_reader = graph_readers.BufferedGraphReader(
        self.db,
        filters=filters,
        order_by_random=True,
        eager_graph_loading=True,
        buffer_size=min(512, FLAGS.batch_size // 10),
        limit=FLAGS.max_instance_count)

    graph_state_dropout = (FLAGS.graph_state_dropout_keep_prob
                               if epoch_type == "train" else 1.0)
    edge_weight_dropout = (FLAGS.edge_weight_dropout_keep_prob
                                      if epoch_type == "train" else 1.0)

    # Pack until we cannot fit more graphs in the batch
    while True:
      with prof.Profile(lambda t: f"Created batch of {humanize.Commas(num_graphs)} graphs"):
        num_graphs = 0
        target_values = []
        adjacency_lists = [[] for _ in range(self.stats.edge_type_count)]
        embedding_indices = [[] for _ in range(self.stats.edge_type_count)]
        incoming_edges = []
        batch_graph_nodes_list = []
        node_offset = 0

        try:
          graph = next(graph_reader)
        except StopIteration:
          # We have reached the end of the database.
          break

        while graph and node_offset + graph.node_count < FLAGS.batch_size:
          # De-serialize pickled data in database and process.
          graph_dict = graph.pickled_data

          batch_graph_nodes_list.append(
              np.full(
                  shape=[graph.node_count],
                  fill_value=num_graphs,
                  dtype=np.int32,
              ))

          # Offset the adjacency list node indices.
          for i in range(self.stats.edge_type_count):
            adjacency_list = graph_dict["adjacency_lists"][i]
            embedding_indices = graph_dict["edge_x"][i]
            if adjacency_list.size:
              adjacency_lists[i].append(adjacency_list + np.array(
                  (node_offset, node_offset), dtype=np.int32))
            if embedding_indices.size:
              embedding_indices[i].append(embedding_indices)

          # Turn counters for incoming edges into a dense array:
          incoming_edges.append(
              graph_dicts.IncomingEdgeCountsToDense(
                  graph_dict["incoming_edge_counts"],
                  node_count=graph.node_count,
                  edge_type_count=self.stats.edge_type_count))

          # TODO(cec): This is hard-coded to classifyapp task.
          #
          # Because classes start counting at 1.
          target_value = graph_dict["graph_y"][0] - 1
          # sanity checks
          assert (target_value <= 103 and
                  target_value >= 0), f"target_value range wrong: {target_value}"
          target_values.append(target_value)

          num_graphs += 1
          node_offset += graph.node_count

          try:
            graph = next(graph_reader)
          except StopIteration:
            # We have reached the end of the database.
            break

        # list[np.array], len ~ 681 (num of graphs in batch)
        incoming_edges = np.concatenate(
            incoming_edges,
            axis=0)
        # list[np.array], len ~ 681 (num of graphs in batch)
        batch_graph_nodes_list = np.concatenate(batch_graph_nodes_list)

        target_values = np.array(target_values)

        batch_feed_dict = {
            self.placeholders["num_incoming_edges_per_type"]: incoming_edges,
            self.placeholders["graph_nodes_list"]: batch_graph_nodes_list,
            self.placeholders["target_values"]: target_values,
            self.placeholders["num_graphs"]: num_graphs,
            self.placeholders["num_of_nodes_in_batch"]: node_offset,
            self.placeholders["graph_state_keep_prob"]: graph_state_dropout,
            self.placeholders["edge_weight_dropout_keep_prob"]: edge_weight_dropout,
        }

        # Merge adjacency lists and information about incoming nodes:
        for i in range(self.stats.edge_type_count):
          if len(adjacency_lists[i]) > 0:
            adj_list = np.concatenate(adjacency_lists[i])
          else:
            adj_list = np.zeros((0, 2), dtype=np.int32)  # shape (0, 2)
          batch_feed_dict[self.placeholders["adjacency_lists"][i]] = adj_list

          if len(embedding_indices[i]) > 0:
            embedding_indices = np.concatenate(embedding_indices[i])
          else:
            embedding_indices = np.zeros((0), dtype=np.int32)  # shape (0, 2)
          batch_feed_dict[self.placeholders["edge_x"][i]] = embedding_indices

      yield num_graphs, batch_feed_dict


def main():
  db = FLAGS.graph_db()
  working_dir = FLAGS.working_dir
  if not working_dir:
    raise app.UsageError("--working_dir is required")

  app.Log(1, 'Using working dir %s', working_dir)

  model = GgnnGraphClassifierModel(db)
  model.Train()


if __name__ == '__main__':
  app.Run(main)
