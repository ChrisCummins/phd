"""Unit tests for //deeplearning/ml4pl/models:run."""
import copy
import random
from typing import Iterable
from typing import Tuple

import numpy as np

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import run
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import progress
from labm8.py import test
from labm8.py.internal import flags_parsers

FLAGS = test.FLAGS


###############################################################################
# Fixtures and mocks.
###############################################################################


@test.Fixture(
  scope="session",
  params=((0, 2), (0, 3), (2, 0), (10, 0)),
  names=("node_y=2", "node_y=3", "graph_y=2", "graph_y=10"),
)
def y_dimensionalities(request) -> Tuple[int, int]:
  """A test fixture which enumerates node and graph label dimensionalities.

  We are interested in testing the following setups:
    * Binary node classification.
    * Multi-class node classification.
    * Binary graph classification.
    * Multi-class graph classification.
  """
  return request.param


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def graph_db(
  request, y_dimensionalities: Tuple[int, int],
) -> graph_tuple_database.Database:
  """A test fixture which enumerates session-level graph databases."""
  node_y_dimensionality, graph_y_dimensionality = y_dimensionalities

  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count=100,
      node_x_dimensionality=2,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=2,
      graph_y_dimensionality=graph_y_dimensionality,
      with_data_flow=False,
      split_count=3,
    )
    yield db


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("log_db"),
)
def disposable_log_db(request) -> log_database.Database:
  """A test fixture which yields an empty log database for every test."""
  yield from testing_databases.YieldDatabase(
    log_database.Database, request.param
  )


class MockModel(classifier_base.ClassifierBase):
  """A mock classifier model."""

  def __init__(
    self, *args, has_loss: bool = True, has_learning_rate: bool = True, **kwargs
  ):
    self.model_data = None
    self.has_loss = has_loss
    self.has_learning_rate = has_learning_rate

    # Counters for testing method calls.
    self.create_model_data_count = 0
    self.make_batch_count = 0
    self.run_batch_count = 0
    self.graph_count = 0

    super(MockModel, self).__init__(*args, **kwargs)

  def CreateModelData(self) -> None:
    """Generate the "state" of the model."""
    self.create_model_data_count += 1
    self.model_data = {"foo": 1, "last_graph_id": None}

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Generate a fake batch of data."""
    graph_ids = []
    while len(graph_ids) < 100:
      try:
        graph_ids.append(next(graphs).id)
      except StopIteration:
        break
    self.make_batch_count += 1
    return batches.Data(graph_ids=graph_ids, data=123)

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batches.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Results:
    # Update the mock class counters.
    self.run_batch_count += 1
    self.graph_count += len(batch.graph_ids)
    self.model_data["last_graph_id"] = batch.graph_ids[-1]

    # Sanity check that batch data is propagated.
    assert batch.data == 123

    learning_rate = random.random() if self.has_learning_rate else None
    loss = random.random() if self.has_loss else None

    results = batches.Results.Create(
      targets=np.random.rand(batch.graph_count, self.y_dimensionality),
      predictions=np.random.rand(batch.graph_count, self.y_dimensionality),
      iteration_count=random.randint(1, 3),
      model_converged=random.choice([False, True]),
      learning_rate=learning_rate,
      loss=loss,
    )

    # Sanity check results properties.
    if self.has_learning_rate:
      assert results.has_learning_rate
    if self.has_loss:
      assert results.has_loss
    return results

  def GetModelData(self):
    """Prepare data to save."""
    return self.model_data

  def LoadModelData(self, data_to_load):
    """Reset model state from loaded data."""
    self.model_data = copy.deepcopy(data_to_load)


###############################################################################
# Tests.
###############################################################################


@test.Parametrize("epoch_count", (1, 5), names=["1_epoch", "5_epochs"])
@test.Parametrize("k_fold", (False, True), names=["one_run", "k_fold"])
@test.Parametrize(
  "run_with_memory_profiler",
  (False, True),
  names=("memprof=false", "memprof=true"),
)
def test_Run_with_mock_module(
  disposable_log_db: log_database.Database,
  graph_db: graph_tuple_database.Database,
  epoch_count: int,
  k_fold: bool,
  run_with_memory_profiler: bool,
):
  """Test the run.Run() method."""
  log_db = disposable_log_db

  # Set the flags that determine the behaviour of Run().
  FLAGS.graph_db = flags_parsers.DatabaseFlag(
    graph_tuple_database.Database, graph_db.url, must_exist=True
  )
  FLAGS.log_db = flags_parsers.DatabaseFlag(
    log_database.Database, log_db.url, must_exist=True
  )
  FLAGS.epoch_count = epoch_count
  FLAGS.k_fold = k_fold
  FLAGS.run_with_memory_profiler = run_with_memory_profiler

  run.Run(MockModel)

  # Test that k-fold produces multiple runs.
  assert log_db.run_count == graph_db.split_count if k_fold else 1


if __name__ == "__main__":
  test.Main()
