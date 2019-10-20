"""Utilities for working with GGNNs."""
import numpy as np
import queue
import tensorflow as tf
import threading

from labm8 import app

FLAGS = app.FLAGS

SMALL_NUMBER = 1e-7


def glorot_init(shape):
  initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
  return np.random.uniform(
      low=-initialization_range, high=initialization_range,
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
  # MLP does just what you would expect: A relu activated, dropout equipped stack of linear layers.
  # If hid=[] then exactly one weight layer W, b is created.
  def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
    self.in_size = in_size
    self.out_size = out_size
    self.hid_sizes = hid_sizes
    self.dropout_keep_prob = dropout_keep_prob
    self.params = self.make_network_params()

  def make_network_params(self):
    dims = [self.in_size] + self.hid_sizes + [self.out_size]
    weight_sizes = list(zip(dims[:-1], dims[1:]))
    weights = [
        tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
        for (i, s) in enumerate(weight_sizes)
    ]
    biases = [
        tf.Variable(
            np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
        for (i, s) in enumerate(weight_sizes)
    ]

    network_params = {
        "weights": weights,
        "biases": biases,
    }

    return network_params

  def init_weights(self, shape):
    return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (
        2 * np.random.rand(*shape).astype(np.float32) - 1)

  def __call__(self, inputs):
    acts = inputs
    for W, b in zip(self.params["weights"], self.params["biases"]):
      hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
      acts = tf.nn.relu(hid)
    last_hidden = hid
    return last_hidden


def GetActivationFunctionFromName(name: str) -> typing.Callable[[typing.Any], tf.Tensor]:
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
