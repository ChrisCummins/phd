"""Unit tests for //deeplearning/ml4pl/models/ggnn"""
import typing

from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.ggnn import ggnn
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


class MockModel(ggnn.Ggnn):
  """A mock GGNN model class."""

  def __init__(self, ggnn_layer_timesteps: typing.List[int] = [30]):
    FLAGS.ggnn_layer_timesteps = ggnn_layer_timesteps


# GetUnrollFactor() tests.


def test_GetUnrollFactor_none():
  model = MockModel()
  log = log_database.BatchLogMeta(type="train")
  assert model.GetUnrollFactor("none", 1, log) == 1

  log = log_database.BatchLogMeta(type="test")
  assert model.GetUnrollFactor("none", 1, log) == 1


def test_GetUnrollFactor_constant():
  model = MockModel()
  log = log_database.BatchLogMeta(type="test")
  assert model.GetUnrollFactor("constant", 2.5, log) == 2
  assert model.GetUnrollFactor("constant", 10, log) == 10

  log = log_database.BatchLogMeta(type="train")
  assert model.GetUnrollFactor("constant", 2.5, log) == 1
  assert model.GetUnrollFactor("constant", 10, log) == 1


def test_GetUnrollFactor_data_flow_max_steps():
  model = MockModel([10])
  log = log_database.BatchLogMeta(type="test")
  log._transient_data = {"data_flow_max_steps_required": 10}
  assert model.GetUnrollFactor("data_flow_max_steps", None, log) == 1

  log._transient_data = {"data_flow_max_steps_required": 50}
  assert model.GetUnrollFactor("data_flow_max_steps", None, log) == 5

  log._transient_data = {"data_flow_max_steps_required": 55}
  assert model.GetUnrollFactor("data_flow_max_steps", None, log) == 6

  log = log_database.BatchLogMeta(type="train")
  log._transient_data = {"data_flow_max_steps_required": 55}
  assert model.GetUnrollFactor("data_flow_max_steps", None, log) == 1


def test_GetUnrollFactor_edge_count():
  model = MockModel([10])
  log = log_database.BatchLogMeta(type="test")
  log._transient_data = {"max_edge_count": 10}
  assert model.GetUnrollFactor("edge_count", 1, log) == 1
  assert model.GetUnrollFactor("edge_count", 0.5, log) == 1

  log._transient_data = {"max_edge_count": 50}
  assert model.GetUnrollFactor("edge_count", 1, log) == 5
  assert model.GetUnrollFactor("edge_count", 0.2, log) == 1

  log._transient_data = {"max_edge_count": 55}
  assert model.GetUnrollFactor("edge_count", 3, log) == 17

  log = log_database.BatchLogMeta(type="train")
  log._transient_data = {"max_edge_count": 55}
  assert model.GetUnrollFactor("edge_count", 3, log) == 1


def test_GetUnrollFactor_label_convergence():
  model = MockModel([10])
  log = log_database.BatchLogMeta(type="train")
  assert model.GetUnrollFactor("label_convergence", None, log) == 1

  log = log_database.BatchLogMeta(type="test")
  assert model.GetUnrollFactor("label_convergence", None, log) == 0


if __name__ == "__main__":
  test.Main()
