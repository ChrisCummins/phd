"""Unit tests for //deeplearning/ml4pl/models/ggnn:ggnn_base."""
import numpy as np
import pathlib
import pickle
import pytest
import tensorflow as tf
import typing

from deeplearning.ml4pl.models.ggnn import ggnn_base
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


class MockModel(ggnn_base.GgnnBaseModel):
  """A mock GGNN model."""

  def MakeLossAndAccuracyAndPredictionOps(self):
    self.placeholders["X"] = tf.placeholder("float")
    self.placeholders["Y"] = tf.placeholder("float")
    W = tf.Variable(np.random.randn(), name="weight")
    b = tf.Variable(np.random.randn(), name="bias")
    predictions = tf.add(tf.multiply(self.placeholders["X"], W), b)
    loss = tf.reduce_sum(tf.pow(predictions - self.placeholders["Y"], 2))
    accuracy = tf.Variable(np.random.randn())
    return loss, accuracy, predictions

  def MakeMinibatchIterator(
      self,
      epoch_type) -> typing.Iterable[typing.Tuple[int, ggnn_base.FeedDict]]:
    del epoch_type
    for _ in range(3):
      yield 1, {
          self.placeholders["X"]: 5,
          self.placeholders["Y"]: 10,
      }


def test_SaveModel(tempdir: pathlib.Path, tempdir2: pathlib.Path):
  """Test saving a model to file."""
  FLAGS.working_dir = tempdir2

  model = MockModel()
  model.global_training_step = 10
  model.SaveModel(tempdir / 'foo.pickle')
  assert (tempdir / 'foo.pickle').is_file()

  with open(tempdir / 'foo.pickle', 'rb') as f:
    saved_model = pickle.load(f)

  assert 'model_flags' in saved_model
  assert 'weights' in saved_model
  assert saved_model['global_training_step'] == 10


def test_LoadModel(tempdir: pathlib.Path, tempdir2: pathlib.Path):
  """Test loading a model from file."""
  FLAGS.working_dir = tempdir2

  model = MockModel()
  model.global_training_step = 10
  model.SaveModel(tempdir / 'foo.pickle')

  model.global_training_step = 0
  model.LoadModel(tempdir / 'foo.pickle')
  assert model.global_training_step == 10


def test_LoadModel_unknown_saved_model_flag(tempdir: pathlib.Path,
                                            tempdir2: pathlib.Path):
  """Test that error is raised if saved model contains unknown flag."""
  FLAGS.working_dir = tempdir2
  model = MockModel()
  model.SaveModel(tempdir / 'foo.pickle')

  with open(tempdir / 'foo.pickle', 'rb') as f:
    saved_model = pickle.load(f)

  saved_model['model_flags']['a new flag'] = 10

  with open(tempdir / 'foo.pickle', 'wb') as f:
    pickle.dump(saved_model, f)

  with pytest.raises(EnvironmentError) as e_ctx:
    model.LoadModel(tempdir / 'foo.pickle')

  assert 'a new flag' in str(e_ctx.value)


def test_Train(tempdir: pathlib.Path, tempdir2: pathlib.Path):
  """Test that training terminates and bumps the epoch number."""
  FLAGS.working_dir = tempdir2
  FLAGS.num_epochs = 1

  model = MockModel()
  model.Train()
  assert model.best_epoch_num == 1


if __name__ == '__main__':
  test.Main()
