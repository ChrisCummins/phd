"""Unit tests for //datasets/mnist/py:mnist_dataset."""
import numpy as np

from datasets.mnist.py import mnist_data_loader
from labm8.py import test


def test_LoadMnistDataset():
  """Test loading the MNIST dataset."""
  loader = mnist_data_loader.MnistDataLoader()
  data = loader.Load()

  assert len(data.train.images) == 60000
  assert len(data.train.labels) == 60000

  assert len(data.test.images) == 10000
  assert len(data.test.labels) == 10000


def test_LoadMnistDataset_numpy():
  """Test loading the MNIST dataset into numpy arrays."""
  loader = mnist_data_loader.MnistDataLoader()
  data = loader.Load()

  train_X = np.array(data.train.images, dtype=np.int8).reshape((60000, 28 * 28))
  train_y = np.array(data.train.labels, dtype=np.int8).reshape((60000, 1))

  assert train_X.shape == ((60000, 28 * 28))
  assert train_y.shape == ((60000, 1))

  test_X = np.array(data.test.images, dtype=np.int8).reshape((10000, 28 * 28))
  test_y = np.array(data.test.labels, dtype=np.int8).reshape((10000, 1))

  assert test_X.shape == ((10000, 28 * 28))
  assert test_y.shape == ((10000, 1))


if __name__ == "__main__":
  test.Main()
