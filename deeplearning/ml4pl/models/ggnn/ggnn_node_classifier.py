"""Train and evaluate a model for reachability analysis.

  $ bazel run //deeplearning/ml4pl/models/ggnn:ggnn_node_classifier -- \
        --graph_db=$DATABASE \
        --batch_size=2000 \
        --max_instance_count=100 \
        --alsologtostderr
"""
import collections
import numpy as np
import pickle
import tensorflow as tf
import typing

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as graph_readers
from deeplearning.ml4pl.graphs import graph_database_stats as graph_stats
from deeplearning.ml4pl.models.ggnn import ggnn_base as ggnn
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils
from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_integer("batch_size", 8000,
                   "The maximum number of nodes to include in a graph batch.")
app.DEFINE_database('graph_db',
                    graph_database.Database,
                    None,
                    'The database to read graph data from.',
                    must_exist=True)
app.DEFINE_integer(
    'max_steps', 0,
    'If > 0, limit the graphs used to those that can be computed in '
    '<= max_steps')
app.DEFINE_integer(
    'max_instance_count', None,
    'A debugging option. Use this to set the maximum number of instances used '
    'from training/validation/test files. Note this still requires reading '
    'the entirety of the file contents into memory.')
app.DEFINE_string("graph_rnn_cell", "GRU",
                  "The RNN cell type. One of {GRU,CudnnCompatibleGRUCell,RNN}")
app.DEFINE_string("graph_rnn_activation", "tanh",
                  "The RNN activation type. One of {tanh,ReLU}")
app.DEFINE_boolean("use_propagation_attention", False, "")
app.DEFINE_boolean("use_edge_bias", False, "")
app.DEFINE_boolean("use_edge_msg_avg_aggregation", True, "")
app.DEFINE_float("graph_state_dropout_keep_prob", 1.0,
                 "Graph state dropout keep probability (rate = 1 - keep_prob)")
app.DEFINE_float("edge_weight_dropout_keep_prob", 1.0,
                 "Edge weight dropout keep probability (rate = 1 - keep_prob)")

# Type aliases to help make sense of the dictionaries.

EdgeType = int  # A categorical value indicating the type of edge. Zero-based.
NodeIndex = int  # A categorical value indicating a node. Zero-based.
# A GraphDict is a dictionary describing the properties of a graph, containing
# various attributes such as an edge list, node count, etc.
GraphDict = typing.Dict[str, typing.Any]
# A source node, edge type, destination node, and embedding index (unused in
# this file).
Edge = typing.Tuple[NodeIndex, EdgeType, NodeIndex, int]
# A list of <source, destination> node pairs.
AdjacencyList = typing.List[typing.Tuple[NodeIndex, NodeIndex]]

GGNNWeights = collections.namedtuple(
    "GGNNWeights",
    [
        "edge_weights",
        "edge_biases",
        "edge_type_attention_weights",
        "rnn_cells",
    ],
)


def ProcessGraphDict(
    packed_args: typing.Tuple[GraphDict, int, bool]) -> GraphDict:
  """Process a graph dictionary by building and setting adjacency lists and
  incoming node counts.

  This replaces the edge list with an adjacency list.

  Adapted from GGNNClassifyappModel.process_raw_graphs() and
  GGNNClassifyappModel.__graph_to_adjacency_lists().

  Changes:
    * Graph dictionaries (as loaded from the train/val/test files) are modified
      in-place. This reduces memory churn and enables this function to be called
      in parallel across the lists of graphs.
    * Replaced the edge type dictionaries with lists. There are only a small
      number of edge types so we can just use lists and index into them by the
      edge type.
    * Removed defaultdict in favour of vanilla dictionaries.
  """
  # Unpack the arguments.
  graph_dict, edge_type_count, insert_backward_edges = packed_args

  # Adjacency lists, one for each edge type.
  adjacency_lists: typing.List[AdjacencyList] = []
  # Lists of incoming edge counts for each mode, one for each edge type.
  incoming_edge_counts_per_type_lists: typing.List[IncomingEdgeCount] = []

  # Initialize the per-edge type lists.
  for _ in range(edge_type_count):
    adjacency_lists.append([])
    incoming_edge_counts_per_type_lists.append({})

  for src, edge_type, dst, unused_embedding_index in graph_dict[
      'adjacency_list']:
    adjacency_lists[edge_type].append((src, dst))
    incoming_edge_counts_per_type_lists[edge_type][dst] = (
        incoming_edge_counts_per_type_lists[edge_type].get(dst, 0) + 1)

    # Optionally insert backward edges.
    if insert_backward_edges:
      adjacency_lists[edge_type].append((dst, src))
      incoming_edge_counts_per_type_lists[edge_type][src] = (
          incoming_edge_counts_per_type_lists[edge_type].get(src, 0) + 1)
      # Add backward edges as an additional edge type that goes backwards.
      backward_edge_type = (edge_type_count // 2) + edge_type
      adjacency_lists[backward_edge_type].append((dst, src))
      incoming_edge_counts_per_type_lists[backward_edge_type][src] = (
          incoming_edge_counts_per_type_lists[backward_edge_type].get(src, 0) +
          1)

  # Sort the adjacency lists and convert to numpy arrays.
  adjacency_lists = [
      np.array(sorted(adjacency_list), dtype=np.int32)
      for adjacency_list in adjacency_lists
  ]

  graph_dict['adjacency_lists'] = adjacency_lists
  graph_dict['num_incoming_edge_per_type'] = incoming_edge_counts_per_type_lists

  # No need to keep the adjacency list any more.
  del graph_dict['adjacency_list']

  return graph_dict


# TODO(cec): Refactor.
residual_connections = {}


class GgnnNodeClassifierModel(ggnn.GgnnBaseModel):
  """GGNN model for learning node classification."""

  def __init__(self, db: graph_database.Database):
    self.db = db
    filters = []
    if FLAGS.max_steps:
      filters.append(lambda: graph_database.GraphMeta.
                     data_flow_max_steps_required <= FLAGS.max_steps)
    self.stats = graph_stats.GraphDatabaseStats(self.db, filters=filters)
    super(GgnnNodeClassifierModel, self).__init__()

  def MakeLossAndAccuracyAndPredictionOps(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Use a single layer GRU, with the minimum number of steps required to
    # process the dataset.
    layer_timesteps = [self.stats.data_flow_max_steps_required]
    app.Log(1, "Using layer timesteps: %s", layer_timesteps)

    self.placeholders["target_values"] = tf.placeholder(tf.int32, [None, 2],
                                                        name="target_values")
    self.placeholders["initial_node_representation"] = tf.placeholder(
        tf.float32, [None, FLAGS.hidden_size], name="node_features")
    self.placeholders["num_of_nodes_in_batch"] = tf.placeholder(
        tf.int32, [], name="num_of_nodes_in_batch")
    adjacency_list_format = ['src', 'dst']
    self.placeholders["adjacency_lists"] = [
        tf.placeholder(tf.int32, [None, len(adjacency_list_format)],
                       name=f"adjacency_e{edge_type}")
        for edge_type in range(self.stats.edge_type_count)
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

    activation_functions = {
        'tanh': tf.nn.tanh,
        'relu': tf.nn.relu,
    }
    activation_function_name = FLAGS.graph_rnn_activation.lower()
    activation_function = activation_functions.get(activation_function_name)
    if not activation_function:
      raise ValueError(
          f"Unknown graph_rnn_activation: {activation_function_name}. "
          f"Allowed values: {list(activation_function.keys())}")

    # Generate per-layer values for edge weights, biases and gated units:
    self.weights = {}  # Used by super-class to place generic things
    self.gnn_weights = GGNNWeights([], [], [], [])
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

        cell_type_name = FLAGS.graph_rnn_cell.lower()
        if cell_type_name == "gru":
          cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size,
                                        activation=activation_function,
                                        name=f"cell_layer_{layer_index}")
        elif cell_type_name == "cudnncompatiblegrucell":
          import tensorflow.contrib.cudnn_rnn as cudnn_rnn
          if activation_function_name != "tanh":
            raise ValueError(
                "cudnncompatiblegrucell must be used with tanh activation")
          cell = cudnn_rnn.CudnnCompatibleGRUCell(FLAGS.hidden_size)
        elif cell_type_name == "rnn":
          cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size,
                                             activation=activation_function)
        else:
          raise ValueError(f"Unknown RNN cell type '{cell_type_name}'.")

        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            cell, state_keep_prob=self.placeholders["graph_state_keep_prob"])
        self.gnn_weights.rnn_cells.append(cell)

    # One entry per layer (final state of that layer), shape: number of nodes
    # in batch v x D.
    node_states_per_layer = []

    # Number of nodes in batch.
    num_nodes_in_batch = self.placeholders['num_of_nodes_in_batch']

    # We drop the initial states placeholder in the first layer as they are all
    # zero.
    node_states_per_layer.append(
        tf.zeros(
            [num_nodes_in_batch, FLAGS.hidden_size],
            dtype=tf.float32,
        ))

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

    for (layer_idx, num_timesteps) in enumerate(layer_timesteps):
      with tf.variable_scope("gnn_layer_%i" % layer_idx):
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        # Extract residual messages, if any:
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
          with tf.variable_scope("timestep_%i" % step):
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
                  self.placeholders["num_incoming_edges_per_type"],
                  self.gnn_weights.edge_biases[layer_idx],
              )  # Shape [V, D]

            if FLAGS.use_edge_msg_avg_aggregation:
              num_incoming_edges = tf.reduce_sum(
                  self.placeholders["num_incoming_edges_per_type"],
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

    self.ops["final_node_representations"] = node_states_per_layer[-1]

    with tf.variable_scope("output_layer"):
      with tf.variable_scope("regression_gate"):
        self.weights["regression_gate"] = utils.MLP(
            # Concatenation of initial and final node states
            in_size=2 * FLAGS.hidden_size,
            out_size=self.stats.node_labels_dimensionality,
            hid_sizes=[],
            dropout_keep_prob=self.placeholders["out_layer_dropout_keep_prob"],
        )

      with tf.variable_scope("regression"):
        self.weights["regression_transform"] = utils.MLP(
            in_size=FLAGS.hidden_size,
            out_size=self.stats.node_labels_dimensionality,
            hid_sizes=[],
            dropout_keep_prob=self.placeholders["out_layer_dropout_keep_prob"],
        )

      # TODO(cec): Can we just stick this in a loop to run inference for more
      # steps?
      computed_values = self.gated_regression(
          self.ops["final_node_representations"],
          self.weights["regression_gate"],
          self.weights["regression_transform"],
      )  # [v, c]

      predictions = tf.argmax(self.placeholders["target_values"],
                              axis=1,
                              output_type=tf.int32)

      accuracy = tf.reduce_mean(
          tf.cast(
              tf.equal(
                  predictions,
                  tf.argmax(computed_values, axis=1, output_type=tf.int32),
              ),
              tf.float32,
          ))

      loss = tf.losses.softmax_cross_entropy(self.placeholders["target_values"],
                                             computed_values)

    return loss, accuracy, predictions

  def gated_regression(self, last_h, regression_gate, regression_transform):
    """Helper function."""
    # last_h: [v x h]
    initial_node_rep = self.placeholders["initial_node_representation"]
    gate_input = tf.concat([last_h, initial_node_rep], axis=-1)  # [v x 2h]

    return tf.nn.sigmoid(
        regression_gate(gate_input)) * regression_transform(last_h)

  def MakeMinibatchIterator(
      self,
      epoch_type: str) -> typing.Iterable[typing.Tuple[int, ggnn.FeedDict]]:
    """Create minibatches by flattening adjacency matrices into a single
    adjacency matrix with multiple disconnected components."""
    # Graphs are keyed into {train,val,test} groups.
    filters = [lambda: graph_database.GraphMeta.group == epoch_type]

    # Optionally limit the number of steps which are required to compute the
    # graphs.
    if FLAGS.max_steps:
      filters.append(lambda: graph_database.GraphMeta.
                     data_flow_max_steps_required <= FLAGS.max_steps)

    graph_reader = graph_readers.BufferedGraphReader(
        self.db,
        filters=filters,
        order_by_random=True,
        eager_graph_loading=True,
        buffer_size=min(256, FLAGS.batch_size // 10),
        limit=FLAGS.max_instance_count)

    state_dropout_keep_prob = (FLAGS.graph_state_dropout_keep_prob
                               if epoch_type == "train" else 1.0)
    edge_weights_dropout_keep_prob = (FLAGS.edge_weight_dropout_keep_prob
                                      if epoch_type == "train" else 1.0)

    # Pack until we cannot fit more graphs in the batch.
    num_graphs = 0
    while True:
      num_graphs_in_batch = 0
      batch_node_features = []
      batch_target_values = []
      batch_adjacency_lists = [[] for _ in range(self.stats.edge_type_count)]
      batch_num_incoming_edges_per_type = []
      batch_graph_nodes_list = []
      node_offset = 0

      graph = next(graph_reader)

      while (graph and node_offset + graph.node_count < FLAGS.batch_size):
        # De-serialize pickled data in database and process.
        graph_dict = pickle.loads(graph.graph.data)
        ProcessGraphDict(
            (graph_dict, self.stats.edge_type_count, FLAGS.tie_fwd_bkwd))

        # Pad node feature vector of size <= hidden_size up to hidden_size so
        # that the size matches embedding dimensionality.
        padded_features = np.pad(
            graph_dict["node_features"],
            ((0, 0),
             (0, FLAGS.hidden_size - self.stats.node_features_dimensionality)),
            "constant",
        )
        # Shape: [num_nodes, node_feature_dim]
        batch_node_features.extend(padded_features)
        # Shape: [num_nodes, num_classes]
        batch_target_values.extend(graph_dict['targets'])
        batch_graph_nodes_list.append(
            np.full(
                shape=[graph.node_count],
                fill_value=num_graphs_in_batch,
                dtype=np.int32,
            ))
        # Offset the adjacency list node indices.
        for i in range(self.stats.edge_type_count):
          adjacency_list = graph_dict["adjacency_lists"][i]
          if adjacency_list.size:
            batch_adjacency_lists[i].append(adjacency_list + np.array(
                (node_offset, node_offset), dtype=np.int32))

        # Turn counters for incoming edges into a dense array:
        batch_num_incoming_edges_per_type.append(
            graph_dict.IncomingEdgeCountsToDense(
                graph_dict["incoming_edge_counts"],
                node_count=graph.node_count,
                edge_type_count=self.stats.edge_type_count))

        num_graphs += 1
        num_graphs_in_batch += 1
        node_offset += graph.node_count

        try:
          graph = next(graph_reader)
        except StopIteration:
          break

      batch_feed_dict = {
          self.placeholders["initial_node_representation"]:
          np.array(batch_node_features
                  ),  # list[np.array], len ~ 100.000 (batch_size)
          self.placeholders["num_incoming_edges_per_type"]:
          np.concatenate(
              batch_num_incoming_edges_per_type,
              axis=0),  # list[np.array], len ~ 681 (num of graphs in batch)
          self.placeholders["graph_nodes_list"]:
          np.concatenate(batch_graph_nodes_list
                        ),  # list[np.array], len ~ 681 (num of graphs in batch)
          self.placeholders["target_values"]:
          np.array(batch_target_values),  # [g * v, c]
          self.placeholders["num_graphs"]:
          num_graphs_in_batch,
          self.placeholders["num_of_nodes_in_batch"]:
          node_offset,
          self.placeholders["graph_state_keep_prob"]:
          state_dropout_keep_prob,
          self.placeholders["edge_weight_dropout_keep_prob"]:
          edge_weights_dropout_keep_prob,
      }

      # Merge adjacency lists and information about incoming nodes:
      for i in range(self.stats.edge_type_count):
        if len(batch_adjacency_lists[i]) > 0:
          adj_list = np.concatenate(batch_adjacency_lists[i])
        else:
          adj_list = np.zeros((0, 2), dtype=np.int32)  # shape (0, 2).
        batch_feed_dict[self.placeholders["adjacency_lists"][i]] = adj_list

      yield num_graphs, batch_feed_dict


def main():
  """Main entry point."""
  db = FLAGS.graph_db()
  working_dir = FLAGS.working_dir
  if not working_dir:
    raise app.UsageError("--working_dir is required")

  app.Log(1, 'Using working dir %s', working_dir)

  model = GgnnNodeClassifierModel(db)
  model.Train()


if __name__ == '__main__':
  app.Run(main)
