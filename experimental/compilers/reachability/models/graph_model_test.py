"""Unit tests for //experimental/compilers/reachability:graph_model."""
import collections
import json
import pathlib

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from absl import flags

from experimental.compilers.reachability.models import graph_model
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='module')
def graph() -> nx.DiGraph:
  """Test fixture which returns a simple graph for use as input or target."""
  g = nx.DiGraph(features=np.ones(2))
  g.add_node(0, features=np.ones(1))
  g.add_node(1, features=np.ones(1))
  g.add_edge(0, 1, features=np.ones(1))
  yield g


@pytest.fixture(scope='module')
def df(graph: nx.DiGraph) -> pd.DataFrame:
  """Dataframe containing a single training, validation, and test entry."""
  return pd.DataFrame([
    {
      'networkx:input_graph': graph.copy(),
      'networkx:target_graph': graph.copy(),
      'split:type': 'training',
      'graphnet:loss_op': 'GlobalsSoftmaxCrossEntropy',
      'graphnet:accuracy_evaluator': 'OneHotGlobals',
    },
    {
      'networkx:input_graph': graph.copy(),
      'networkx:target_graph': graph.copy(),
      'split:type': 'validation',
      'graphnet:loss_op': 'GlobalsSoftmaxCrossEntropy',
      'graphnet:accuracy_evaluator': 'OneHotGlobals',
    },
    {
      'networkx:input_graph': graph.copy(),
      'networkx:target_graph': graph.copy(),
      'split:type': 'test',
      'graphnet:loss_op': 'GlobalsSoftmaxCrossEntropy',
      'graphnet:accuracy_evaluator': 'OneHotGlobals',
    },
  ])


# A model which has been evaluated.
TrainedModel = collections.namedtuple('TrainedModel', ['model', 'outputs'])


@pytest.fixture(scope='module')
def trained_model(df: pd.DataFrame,
                  module_tempdir: pathlib.Path) -> TrainedModel:
  """Test fixture which yields a trained model and its outputs.'"""
  model = graph_model.CompilerGraphNeuralNetwork(df, module_tempdir)
  with tf.Session() as sess:
    outputs = model.TrainAndEvaluate(sess)
  yield TrainedModel(model, outputs)


def test_CompilerGraphNeuralNetwork_TrainAndEvaluate_tensorboard_files(
    trained_model: TrainedModel):
  """Test that tensorboard files are produced."""
  assert (trained_model.model.outdir / 'tensorboard').is_dir()

  # Check for a single instance of the training / validation / testing
  # tensorboard events file.
  def DirectoryContainsTensorboardEventsFile(path: pathlib.Path):
    assert (path).is_dir()
    tensorboard_files = list((path).iterdir())
    assert len(tensorboard_files) == 1
    assert tensorboard_files[0].name.startswith('events.out.tfevents.')

  DirectoryContainsTensorboardEventsFile(
      trained_model.model.outdir / 'tensorboard/training')
  DirectoryContainsTensorboardEventsFile(
      trained_model.model.outdir / 'tensorboard/validation')
  DirectoryContainsTensorboardEventsFile(
      trained_model.model.outdir / 'tensorboard/test')


def test_CompilerGraphNeuralNetwork_TrainAndEvaluate_telemetry_files(
    trained_model: TrainedModel):
  """Test that telemetry files are produced."""

  assert (trained_model.model.outdir / 'telemetry').is_dir()
  telemetry_files = list((trained_model.model.outdir / 'telemetry').iterdir())
  # There should be one telemetry file per epoch.
  # FIXME(cec): A test shouldn't depend on a flag value!
  assert len(telemetry_files) == FLAGS.num_epochs

  # Verify telemetry files.
  for path in telemetry_files:
    assert path.name.startswith('epoch_')
    assert path.name.endswith('.json')
    # Check that telemetry can be loaded as JSON.
    with open(path) as f:
      telemetry = json.load(f)
      assert telemetry

      # Check for some (NOT ALL!) of the expected valyes.
      assert 'test_outputs' in telemetry
      assert 'test_accuracy' in telemetry
      assert 'training_accuracy' in telemetry
      assert 'validation_accuracy' in telemetry
      assert 'epoch' in telemetry
      assert 'test_loss' in telemetry
      assert 'training_losses' in telemetry
      assert 'validation_loss' in telemetry


if __name__ == '__main__':
  test.Main()
