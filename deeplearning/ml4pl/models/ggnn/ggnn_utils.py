"""Utilities for working with GGNNs."""
import typing

import numpy as np
import tensorflow as tf
from labm8 import app

from deeplearning.ml4pl.graphs import graph_database_stats

FLAGS = app.FLAGS

SMALL_NUMBER = 1e-7


def glorot_init(shape):
  initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
  return np.random.uniform(low=-initialization_range,
                           high=initialization_range,
                           size=shape).astype(np.float32)


def uniform_init(shape):
  return np.random.uniform(low=-1.0, high=1, size=shape).astype(np.float32)


class MLP(object):
  """MLP does just what you would expect: A relu activated, dropout equipped
  stack of linear layers.
  """

  def __init__(self, in_size: int, out_size: int,
               hidden_sizes: typing.List[int],
               dropout_keep_prob: typing.Optional[tf.Tensor]):
    """Constructor.

    If hidden_sizes=[] then exactly one weight layer W, b is created.
    """
    self.in_size = in_size
    self.out_size = out_size
    self.hidden_sizes = hidden_sizes
    self.dropout_keep_prob = dropout_keep_prob
    self.params = self.MakeNetworkParameters()

  def MakeNetworkParameters(self):
    dims = [self.in_size] + self.hidden_sizes + [self.out_size]
    weight_sizes = list(zip(dims[:-1], dims[1:]))
    weights = [
        tf.Variable(self.InitialWeights(s), name='MLP_W_layer%i' % i)
        for (i, s) in enumerate(weight_sizes)
    ]
    biases = [
        tf.Variable(np.zeros(s[-1]).astype(np.float32),
                    name='MLP_b_layer%i' % i)
        for (i, s) in enumerate(weight_sizes)
    ]

    network_params = {
        "weights": weights,
        "biases": biases,
    }

    return network_params

  def InitialWeights(self, shape):
    return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (
        2 * np.random.rand(*shape).astype(np.float32) - 1)

  def __call__(self, inputs):
    acts = inputs
    for W, b in zip(self.params["weights"], self.params["biases"]):
      if self.dropout_keep_prob is not None:
        W = tf.nn.dropout(W, self.dropout_keep_prob)
      hid = tf.matmul(acts, W) + b
      acts = tf.nn.relu(hid)
    last_hidden = hid
    return last_hidden


ActivationFunction = typing.Callable[[tf.Tensor], tf.Tensor]


def GetActivationFunctionFromName(name: str) -> ActivationFunction:
  activation_functions = {
      'tanh': tf.nn.tanh,
      'relu': tf.nn.relu,
  }
  activation_function = activation_functions.get(name.lower())
  if not activation_function:
    raise ValueError(
        f"Unknown activation function: `{activation_function_name}`. "
        f"Allowed values: {list(activation_function.keys())}")
  return activation_function


def BuildRnnCell(cell_type: str,
                 activation_function: str,
                 hidden_size: int,
                 name: typing.Optional[str] = None):
  activation_function = GetActivationFunctionFromName(activation_function)

  cell_type = cell_type.lower()
  if cell_type == "gru":
    return tf.nn.rnn_cell.GRUCell(hidden_size,
                                  activation=activation_function,
                                  name=name)
  elif cell_type == "cudnncompatiblegrucell":
    import tensorflow.contrib.cudnn_rnn as cudnn_rnn
    if activation_function != tf.nn.tanh:
      raise ValueError(
          "cudnncompatiblegrucell must be used with tanh activation")
    return cudnn_rnn.CudnnCompatibleGRUCell(hidden_size, name=name)
  elif cell_type == "rnn":
    return tf.nn.rnn_cell.BasicRNNCell(hidden_size,
                                       activation=activation_function,
                                       name=name)
  else:
    raise ValueError(f"Unknown RNN cell type '{name}'.")


def MakePlaceholders(stats: graph_database_stats.GraphTupleDatabaseStats
                    ) -> typing.Dict[str, tf.Tensor]:
  """Create tensorflow placeholders for graph tuples in the given dataset.

  Args:
    stats: A graph stats object which provides access to properties describing
      the graph tuples.

  Returns:
    A dictionary mapping names to placeholder tensors.
  """
  placeholders = {
      'graph_count':
      tf.compat.v1.placeholder(tf.int32, [], name="graph_count"),
      'node_count':
      tf.compat.v1.placeholder(tf.int32, [], name="node_count"),
      'adjacency_lists': [
          tf.compat.v1.placeholder(tf.int32, [None, 2], name=f"adjacency_e{i}")
          for i in range(stats.edge_type_count)
      ],
      "edge_positions": [
          tf.compat.v1.placeholder(dtype=tf.int32,
                                   shape=[None],
                                   name=f'edge_positions_e{i}')
          for i in range(stats.edge_type_count)
      ],
      'incoming_edge_counts':
      tf.compat.v1.placeholder(tf.float32, [None, stats.edge_type_count],
                               name="incoming_edge_counts"),
      'graph_nodes_list':
      tf.compat.v1.placeholder(tf.int32, [None], name="graph_nodes_list"),
      # Dropouts:
      'graph_state_dropout_keep_prob':
      tf.compat.v1.placeholder(tf.float32, None, name="graph_state_keep_prob"),
      'edge_weight_dropout_keep_prob':
      tf.compat.v1.placeholder(tf.float32,
                               None,
                               name="edge_weight_dropout_keep_prob"),
      "output_layer_dropout_keep_prob":
      tf.compat.v1.placeholder(tf.float32, [],
                               name="output_layer_dropout_keep_prob"),
      "is_training":
      tf.compat.v1.placeholder(dtype=tf.bool, shape=[], name='is_training'),
      "learning_rate_multiple":
      tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name='learning_rate_multiple')
  }

  # This is a list of node embedding table indices. Each row is a node, column
  # {i} is an index into embedding table {i}.
  placeholders['node_x'] = tf.compat.v1.placeholder(
      dtype=tf.int32, shape=[None, stats.node_embeddings_count], name="node_x")

  placeholders['raw_node_output_features'] = tf.compat.v1.placeholder(
      stats.node_embeddings_dtype,
      [None, stats.node_embeddings_concatenated_width],
      name="raw_node_output_features")

  # TODO(cec): Is there ever a case where --hiden_size does not need to equal
  # node_embeddings_concatenated_width? If not, then lets remove the hidden_size
  # flag and instead derive it.
  if FLAGS.hidden_size != stats.node_embeddings_concatenated_width:
    raise ValueError(f"--hidden_size={FLAGS.hidden_size} != "
                     "node_embeddings_concatenated_width="
                     f"{stats.node_embeddings_concatenated_width}")

  if stats.node_labels_dimensionality:
    placeholders['node_y'] = tf.compat.v1.placeholder(
        stats.node_labels_dtype, [None, stats.node_labels_dimensionality],
        name="node_y")

  if stats.graph_features_dimensionality:
    placeholders['graph_x'] = tf.compat.v1.placeholder(
        stats.graph_features_dtype, [None, stats.graph_features_dimensionality],
        name="graph_x")

  if stats.graph_labels_dimensionality:
    placeholders['graph_y'] = tf.compat.v1.placeholder(
        #stats.graph_labels_dtype, [None, stats.graph_labels_dimensionality],
        stats.graph_labels_dtype,
        [None, 2],
        name="graph_y")

  return placeholders

def BatchDictToFeedDict(
    batch: typing.Dict[str, typing.Any],
    placeholders: typing.Dict[str, tf.Tensor],
) -> typing.Dict[tf.Tensor, typing.Any]:
  """Re-key a batch dictionary to use the given placeholder names.

  Args:
    batch: A batch dictionary as produced by the
      GraphBatcher.MakeGraphBatchIterator() iterator.
    placeholders: A dictionary mapping placeholder names to the names returned
      by tf.compat.v1.placeholder().

  Returns:
    The batch dictionary values, re-keyed by the corresponding values in the
    placeholders dictionary.
  """
  edge_type_count = len(batch.adjacency_lists)

  feed_dict = {
      placeholders["graph_count"]: batch.graph_count,
      placeholders['graph_nodes_list']: batch.graph_nodes_list,
      placeholders["node_x"]: batch.node_x_indices,
      placeholders["node_count"]: batch.node_count,
      placeholders["incoming_edge_counts"]: batch.incoming_edge_counts,
  }

  for i in range(edge_type_count):
    feed_dict[placeholders["adjacency_lists"][i]] = batch.adjacency_lists[i]
    feed_dict[placeholders["edge_positions"][i]] = batch.edge_positions[i]

  if batch.has_node_y:
    feed_dict[placeholders["node_y"]] = batch.node_y

  if batch.has_graph_x:
    feed_dict[placeholders["graph_x"]] = batch.graph_x

  if batch.has_graph_y:
    feed_dict[placeholders["graph_y"]] = batch.graph_y

  assert placeholders['learning_rate_multiple'] is not None, 'learning_rate_multiple is None'

  return feed_dict


def MakeModularOutputLayer(initial_node_state, final_node_state,
                           regression_gate, regression_transform):
  """An output layer that reuses weights, but can be fed with a placeholder."""
  with tf.compat.v1.variable_scope("modular_output_layer"):
    with tf.compat.v1.variable_scope("gated_regression"):
      gated_input = tf.concat([final_node_state, initial_node_state], axis=-1)

      computed_values = (tf.nn.sigmoid(regression_gate(gated_input)) *
                         regression_transform(final_node_state))
  return computed_values


def MakeOutputLayer(initial_node_state, final_node_state, hidden_size: int,
                    labels_dimensionality: int,
                    dropout_keep_prob_placeholder: typing.Optional[str]):
  with tf.compat.v1.variable_scope("output_layer"):
    with tf.compat.v1.variable_scope("regression_gate"):
      regression_gate = MLP(
          # Concatenation of initial and final node states
          in_size=2 * hidden_size,
          out_size=labels_dimensionality,
          hidden_sizes=[],
          dropout_keep_prob=dropout_keep_prob_placeholder,
      )

    with tf.compat.v1.variable_scope("regression_transform"):
      regression_transform = MLP(
          in_size=hidden_size,
          out_size=labels_dimensionality,
          hidden_sizes=[],
          dropout_keep_prob=dropout_keep_prob_placeholder,
      )

    with tf.compat.v1.variable_scope("gated_regression"):
      gated_input = tf.concat([final_node_state, initial_node_state], axis=-1)

      computed_values = (tf.nn.sigmoid(regression_gate(gated_input)) *
                         regression_transform(final_node_state))

  return computed_values, regression_gate, regression_transform


def RunWithFetchDict(sess: tf.compat.v1.Session,
                     fetch_dict: typing.Dict[str, tf.Tensor],
                     feed_dict: typing.Dict[tf.Tensor, typing.Any]
                    ) -> typing.Dict[str, tf.Tensor]:
  """A wrapper around session run which uses a dictionary for the fetch list."""
  fetch_dict_keys = sorted(fetch_dict.keys())
  fetch_dict_values = [fetch_dict[k] for k in fetch_dict_keys]
  values = sess.run(fetch_dict_values, feed_dict)
  return {fetch: value for fetch, value in zip(fetch_dict_keys, values)}
