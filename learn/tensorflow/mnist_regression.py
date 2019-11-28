"""MNIST regression with TensorFlow.

Softmax regression over MNIST dataset, with a Nelder-Mead optimization across
training hyperparameters.
"""
import typing

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tensorflow.examples.tutorials.mnist import input_data

from labm8.py import app

FLAGS = app.FLAGS
app.DEFINE_integer(
  "maxiter", 10, "Maximum number of steps when sweeping hyperparms."
)


def SoftmaxRegressor(tensor_size):
  """Build the regressor: y = Wx + b."""
  # Our tensor shape is N * flattened image size, where N is the
  # number of images:
  x = tf.compat.v1.placeholder(tf.float32, [None, tensor_size])

  # Weights matrix. One weight for each class:
  W = tf.Variable(tf.zeros([tensor_size, 10]))

  # Class biases vector:
  b = tf.Variable(tf.zeros([10]))

  # Model: y = Wx + b
  y = tf.nn.softmax(tf.matmul(x, W) + b)

  # Correct answers:
  y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

  # Cross entropy: -sum(y' * log(y))
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

  # Use Gradient Descent to train:
  # TODO: use a tensorflow variable for learning rate, and add to
  # training hyperparameters.
  learning_rate = 0.01
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  # argmax() to find predicted and true classes:
  predicted = tf.argmax(y, 1)
  actual = tf.argmax(y_, 1)

  # List of booleans:
  correct_prediction = tf.equal(predicted, actual)

  # Return model:
  return {
    "inputs": [x, y_],
    "train": optimizer.minimize(cross_entropy),
    "eval": tf.reduce_mean(tf.cast(correct_prediction, tf.float32)),
  }


def TrainAndTest(
  session, model, training_data, test_data, batch_size=100, num_iterations=1000
) -> float:
  """Train and test regression model."""
  # Initialize variables:
  init = tf.initialize_all_variables()

  session(init)
  for i in range(num_iterations):
    batch = training_data.next_batch(batch_size)
    feed_dict = dict(zip(model["inputs"], batch))
    session(model["train"], feed_dict=feed_dict)

  # Evaluate on test set:
  feed_dict = dict(zip(model["inputs"], [test_data.images, test_data.labels]))
  error = 1 - session(model["eval"], feed_dict=feed_dict)

  app.Log(
    1,
    "MNIST with batch size %d for %d iterations: %.3f %% error ",
    batch_size,
    num_iterations,
    error * 100,
  )
  return error


def Denormalize(inputs):
  """Scale hyperparameters to "real" values."""
  batch_size, num_iterations = inputs
  return int(round(batch_size * 100)), int(round(num_iterations * 1000))


def HyperParamSweep(maxiter: int = 50) -> typing.Dict[str, int]:
  """Perform a hyper-parameter sweep to find the best batch size and numiter."""
  app.Log(1, "loading dataset ... ")
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  image_width = 28  # MNIST image dimensions
  tensor_size = image_width * image_width  # Flatten image to 1D

  # Build model:
  session = tf.compat.v1.Session()
  model = SoftmaxRegressor(tensor_size)

  def f(X):
    """
    Optimization function to train and evaluate model with
    hyperparameters.
    """
    batch_size, num_iterations = Denormalize(X)
    return TrainAndTest(
      session.run,
      model,
      mnist.train,
      mnist.test,
      batch_size=batch_size,
      num_iterations=num_iterations,
    )

  # Hyper-parameter search over batch size and training iterations.
  x0 = np.array([1, 1])
  res = minimize(f, x0, method="nelder-mead", options={"maxiter": maxiter})

  batch_size, numiter = Denormalize(res.x)
  return {"batch_size": batch_size, "numiter": numiter}


def main(argv):
  del argv

  params = HyperParamSweep(FLAGS.maxiter)
  app.Log(
    1,
    "batch size: %d, nuber of iterations: %d",
    params["batch_size"],
    params["numiter"],
  )


if __name__ == "__main__":
  app.RunWithArgs(main)
