"""Utilities for working with GGNNs."""
import numpy as np
import queue
import tensorflow as tf
import threading
import typing

from deeplearning.ml4pl.graphs import graph_database_stats
from labm8 import app

FLAGS = app.FLAGS

SMALL_NUMBER = 1e-7


def glorot_init(shape):
  initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
  return np.random.uniform(low=-initialization_range,
                           high=initialization_range,
                           size=shape).astype(np.float32)


def uniform_init(shape):
  return np.random.uniform(low=-1.0, high=1, size=shape).astype(np.float32)


class ThreadedIterator:
  """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
  The iterator should *not* return None"""

  def __init__(self, original_iterator, max_queue_size: int = 2):
    self.__queue = queue.Queue(maxsize=max_queue_size)
    self.__thread = threading.Thread(
        target=lambda: self.worker(original_iterator))
    self.__thread.start()

  def worker(self, original_iterator):
    for element in original_iterator:
      assert element is not None, 'By convention, iterator elements much not be None'
      self.__queue.put(element, block=True)
    self.__queue.put(None, block=True)

  def __iter__(self):
    next_element = self.__queue.get(block=True)
    while next_element is not None:
      yield next_element
      next_element = self.__queue.get(block=True)
    self.__thread.join()


class MLP(object):
  """MLP does just what you would expect: A relu activated, dropout equipped
  stack of linear layers.
  """

  def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
    """Constructors.

    If hid_sizes=[] then exactly one weight layer W, b is created.
    """
    self.in_size = in_size
    self.out_size = out_size
    self.hid_sizes = hid_sizes
    self.dropout_keep_prob = dropout_keep_prob
    self.params = self.MakeNetworkParameters()

  def MakeNetworkParameters(self):
    dims = [self.in_size] + self.hid_sizes + [self.out_size]
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
      hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
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


def MakePlaceholders(stats: graph_database_stats.GraphDictDatabaseStats
                    ) -> typing.Dict[str, tf.Tensor]:
  """Create tensorflow placeholders for graph dicts in the given dataset.

  Args:
    stats: A graph stats object which provides access to properties describing
      the graph dicts.

  Returns:
    A dictionary mapping names to placeholder tensors.
  """
  placeholders = {
      'graph_count':
      tf.placeholder(tf.int32, [], name="graph_count"),
      'node_count':
      tf.placeholder(tf.int32, [], name="node_count"),
      'adjacency_lists': [
          tf.placeholder(tf.int32, [None, 2], name=f"adjacency_e{i}")
          for i in range(stats.edge_type_count)
      ],
      'incoming_edge_counts':
      tf.placeholder(tf.float32, [None, stats.edge_type_count],
                     name="incoming_edge_counts"),
      'graph_nodes_list':
      tf.placeholder(tf.int32, [None], name="graph_nodes_list"),
      # Dropouts:
      'graph_state_keep_prob':
      tf.placeholder(tf.float32, None, name="graph_state_keep_prob"),
      'edge_weight_dropout_keep_prob':
      tf.placeholder(tf.float32, None, name="edge_weight_dropout_keep_prob"),
      "out_layer_dropout_keep_prob":
      tf.placeholder(tf.float32, [], name="out_layer_dropout_keep_prob"),
  }

  if stats.node_features_dimensionality:
    placeholders['node_x'] = tf.placeholder(
        stats.node_features_dtype,
        # TODO(cec): This is hardcoded to padded node features.
        # It should be stats.node_features_dimensionality.
        [None, FLAGS.hidden_size],
        name="node_x")

  if stats.node_labels_dimensionality:
    placeholders['node_y'] = tf.placeholder(
        stats.node_labels_dtype, [None, stats.node_labels_dimensionality],
        name="node_y")

  if stats.edge_features_dimensionality:
    placeholders['edge_x'] = [
        tf.placeholder(stats.edge_features_dtype,
                       [None, stats.edge_features_dimensionality],
                       name=f"edge_x_e{i}")
        for i in range(stats.edge_type_count)
    ]

  if stats.edge_labels_dimensionality:
    placeholders['edge_x'] = [
        tf.placeholder(stats.edge_labels_dtype,
                       [None, stats.edge_features_dimensionality],
                       name=f"edge_x_e{i}")
        for i in range(stats.edge_type_count)
    ]

  if stats.graph_features_dimensionality:
    placeholders['graph_x'] = tf.placeholder(
        stats.graph_features_dtype, [None, stats.graph_features_dimensionality],
        name="graph_x")

  if stats.graph_labels_dimensionality:
    placeholders['graph_y'] = tf.placeholder(
        stats.graph_labels_dtype, [None, stats.graph_labels_dimensionality],
        name="graph_y")

  return placeholders


def BatchDictToFeedDict(
    batch: typing.Dict[str, typing.Any],
    placeholders: typing.Dict[str, tf.Tensor],
) -> typing.Dict[tf.Tensor, typing.Any]:
  """Re-key a batch dictionary to use the given placeholder names.

  Args:
    batch: A batch dictionary as produced by the
      GraphBatcher.MakeGroupBatchIterator() iterator.
    placeholders: A dictionary mapping placeholder names to the names returned
      by tf.placeholder().

  Returns:
    The batch dictionary values, re-keyed by the corresponding values in the
    placeholders dictionary.
  """
  edge_type_count = len(batch['adjacency_lists'])

  feed_dict = {
      placeholders["incoming_edge_counts"]: batch['incoming_edge_counts'],
      placeholders['graph_nodes_list']: batch['graph_nodes_list'],
      placeholders["graph_count"]: batch['graph_count'],
      placeholders["node_count"]: batch['node_count'],
  }

  for i in range(edge_type_count):
    feed_dict[placeholders["adjacency_lists"][i]] = batch['adjacency_lists'][i]

  if 'node_x' in batch:
    feed_dict[placeholders["node_x"]] = batch['node_x']

  if 'node_y' in batch:
    feed_dict[placeholders["node_y"]] = batch['node_y']

  if 'edge_x' in batch:
    for i in range(edge_type_count):
      feed_dict[placeholders["edge_x"][i]] = batch['edge_x'][i]

  if 'edge_y' in batch:
    for i in range(edge_type_count):
      feed_dict[placeholders["edge_y"][i]] = batch['edge_y'][i]

  if 'graph_x' in batch:
    feed_dict[placeholders["graph_x"]] = batch['graph_x']

  if 'graph_y' in batch:
    feed_dict[placeholders["graph_y"]] = batch['graph_y']

  return feed_dict


def MakeOutputLayer(initial_node_state, final_node_state, hidden_size: int,
                    labels_dimensionality: int,
                    dropout_keep_prob_placeholder: str):
  with tf.variable_scope("output_layer"):
    with tf.variable_scope("regression_gate"):
      regression_gate = MLP(
          # Concatenation of initial and final node states
          in_size=2 * hidden_size,
          out_size=labels_dimensionality,
          hid_sizes=[],
          dropout_keep_prob=dropout_keep_prob_placeholder,
      )

    with tf.variable_scope("regression_transform"):
      regression_transform = MLP(
          in_size=hidden_size,
          out_size=labels_dimensionality,
          hid_sizes=[],
          dropout_keep_prob=dropout_keep_prob_placeholder,
      )

    with tf.variable_scope("gated_regression"):
      gated_input = tf.concat([final_node_state, initial_node_state], axis=-1)

      computed_values = (tf.nn.sigmoid(regression_gate(gated_input)) *
                         regression_transform(initial_node_state))

  return computed_values, regression_gate, regression_transform


def RunWithFetchDict(sess: tf.Session, fetch_dict: typing.Dict[str, tf.Tensor],
                     feed_dict: typing.Dict[tf.Tensor, typing.Any]
                    ) -> typing.Dict[str, tf.Tensor]:
  """A wrapper around session run which uses a dictionary for the fetch list."""
  fetch_dict_keys = sorted(fetch_dict.keys())
  fetch_dict_values = [fetch_dict[k] for k in fetch_dict_keys]
  values = sess.run(fetch_dict_values, feed_dict)
  return {fetch: value for fetch, value in zip(fetch_dict_keys, values)}


def BuildConfusionMatrix(targets: np.array, predictions: np.array) -> np.array:
  """Build a confusion matrix.

  Args:
    targets: A list of 1-hot vectors with shape [num_instances,num_classes].
    predictions: A list of 1-hot vectors with shape [num_instances,num_classes].

  Returns:
    A pickled confusion matrix, which is a matrix of shape
    [num_classes, num_classes] where the rows indicate true target class,
    the columns indicate predicted target class, and the element values are
    the number of instances of this type in the batch.
  """
  num_classes = len(targets[0])

  # Convert 1-hot vectors to dense lists of integers.
  targets = np.argmax(targets, axis=1)
  predictions = np.argmax(predictions, axis=1)

  confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
  for target, prediction in zip(targets, predictions):
    confusion_matrix[target][prediction] += 1

  return confusion_matrix
