#!/usr/bin/env python3
#
# Softmax regression over MNIST dataset, with a Nelder-Mead
# optimization across training hyperparameters.
#
# Uses TensorFlow, scipy, numpy. On Ubuntu install with:
#
#    sudo pip3 install --upgrade scipy numpy https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp34-cp34m-linux_x86_64.whl
#
import sys

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tensorflow.examples.tutorials.mnist import input_data


def softmax_regressor(tensor_size):
  """ Build the regressor: y = Wx + b. """
  # Our tensor shape is N * flattened image size, where N is the
  # number of images:
  x = tf.placeholder(tf.float32, [None, tensor_size])

  # Weights matrix. One weight for each class:
  W = tf.Variable(tf.zeros([tensor_size, 10]))

  # Class biases vector:
  b = tf.Variable(tf.zeros([10]))

  # Model: y = Wx + b
  y = tf.nn.softmax(tf.matmul(x, W) + b)

  # Correct answers:
  y_ = tf.placeholder(tf.float32, [None, 10])

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
    "eval": tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  }


def train_and_test(session, model, training_data, test_data,
                   batch_size=100, num_iterations=1000):
  """ Train and test regression model. """

  print("training with batch size {} for {} iterations ... "
        .format(batch_size, num_iterations), end="")
  sys.stdout.flush()

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
  print("{:.2f}% classification error".format(error * 100))
  return error


def denormalize(inputs):
  """ Scale hypeparameters to "real" values. """
  batch_size, num_iterations = inputs
  return int(round(batch_size * 100)), int(round(num_iterations * 1000))


def main():
  # Load training data
  print("loading dataset ... ")
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  image_width = 28  # MNIST image dimensions
  tensor_size = image_width * image_width  # Flatten image to 1D

  # Build model:
  session = tf.Session()
  model = softmax_regressor(tensor_size)

  def f(X):
    """
    Optimization function to train and evaluate model with
    hyperparameters.
    """
    batch_size, num_iterations = denormalize(X)
    return train_and_test(session.run, model, mnist.train,
                          mnist.test, batch_size=batch_size,
                          num_iterations=num_iterations)

  # Hyper-parameter search over batch size and training iterations.
  maxiter = 50
  x0 = np.array([1, 1])
  res = minimize(f, x0, method="nelder-mead", options={"maxiter": maxiter})

  print("done. batch size: {}, number of iterations: {}"
        .format(*denormalize(res.x)))


if __name__ == "__main__":
  main()
