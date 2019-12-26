# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //deeplearning/ml4pl/models:classifier_base."""
import copy
import random
from typing import Iterable
from typing import Tuple

import numpy as np

from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logging
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import progress
from labm8.py import test


FLAGS = test.FLAGS


###############################################################################
# Fixtures and mocks.
###############################################################################


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("log_db"),
)
def log_db(request) -> log_database.Database:
  """A test fixture which yields an empty log database."""
  yield from testing_databases.YieldDatabase(
    log_database.Database, request.param
  )


@test.Fixture(scope="session")
def logger(log_db: log_database.Database) -> logging.Logger:
  """A test fixture which yields a logger."""
  with logging.Logger(log_db, max_buffer_length=128) as logger:
    yield logger


# Currently, only 2-dimension node features are supported.
@test.Fixture(scope="session", params=(2,), namer=lambda x: f"node_x:{x}")
def node_x_dimensionality(request) -> int:
  """A test fixture which enumerates node feature dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(0, 2), namer=lambda x: f"graph_x:{x}")
def graph_x_dimensionality(request) -> int:
  """A test fixture which enumerates graph feature dimensionalities."""
  return request.param


@test.Fixture(
  scope="session",
  params=((0, 2), (0, 3), (2, 0), (10, 0)),
  namer=lambda x: f"graph_y:{x[0]}-node_y:{x[1]}",
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
  request,
  node_x_dimensionality: int,
  graph_x_dimensionality: int,
  y_dimensionalities: Tuple[int, int],
) -> graph_tuple_database.Database:
  """A test fixture which enumerates graph databases."""
  node_y_dimensionality, graph_y_dimensionality = y_dimensionalities

  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count=50,
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      with_data_flow=True,
    )
    yield db


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
    epoch_type: epoch.Type,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Generate a fake batch of data."""
    del epoch_type  # Unused.
    del ctx  # Unused.

    self.make_batch_count += 1

    graph_ids = []
    while len(graph_ids) < 100:
      try:
        graph_ids.append(next(graphs).id)
      except StopIteration:
        if len(graph_ids) == 0:
          return batches.EndOfBatches()
        break

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


@test.Fixture(
  scope="session", params=(False, True), names=("no_loss", "with_loss")
)
def has_loss(request) -> bool:
  """A test fixture which enumerates losses."""
  return request.param


@test.Fixture(scope="session", params=(False, True), names=("no_lr", "with_lr"))
def has_learning_rate(request) -> bool:
  """A test fixture which enumerates learning rates."""
  return request.param


@test.Fixture(scope="function")
def model(
  logger: logging.Logger,
  graph_db: graph_tuple_database.Database,
  has_loss: bool,
  has_learning_rate: bool,
) -> MockModel:
  """A test fixture which enumerates mock models."""
  return MockModel(
    logger=logger,
    graph_db=graph_db,
    has_loss=has_loss,
    has_learning_rate=has_learning_rate,
  )


@test.Fixture(scope="function")
def batch_iterator(
  model: MockModel, graph_db: graph_tuple_database.Database,
) -> batches.BatchIterator:
  return batches.BatchIterator(
    batches=model.BatchIterator(
      epoch.Type.TRAIN, graph_database_reader.BufferedGraphReader(graph_db)
    ),
    graph_count=graph_db.graph_count,
  )


@test.Fixture(
  scope="session", params=list(epoch.Type), namer=lambda x: x.name.lower()
)
def epoch_type(request) -> epoch.Type:
  """A test fixture which enumerates epoch types."""
  return request.param


@test.Fixture(scope="session", params=(0, 5), namer=lambda x: f"epoch_num:{x}")
def epoch_num(request) -> int:
  return request.param


###############################################################################
# Tests.
###############################################################################


def test_load_restore_model_from_checkpoint(
  logger: logging.Logger,
  graph_db: graph_tuple_database.Database,
  has_loss: bool,
  has_learning_rate: bool,
):
  """Test creating and restoring model from checkpoint."""
  model = MockModel(logger, graph_db)
  model.Initialize()

  # Mutate model state.
  model.epoch_num = 5
  model.model_data = 15

  # Create the checkpoint to restore from.
  checkpoint_ref = model.SaveCheckpoint()
  assert checkpoint_ref.run_id == model.run_id
  assert checkpoint_ref.epoch_num == model.epoch_num

  # Create a new model from this checkpoint.
  restored_model = MockModel(
    logger, graph_db, has_loss=has_loss, has_learning_rate=has_learning_rate,
  )
  restored_model.RestoreFrom(checkpoint_ref)

  # The first model was initialized with Initialize(), the second was not.
  assert model.create_model_data_count == 1
  assert restored_model.create_model_data_count == 0

  # restored_from is a checkpoint reference
  assert isinstance(
    restored_model.restored_from, checkpoints.CheckpointReference
  )
  assert restored_model.restored_from.run_id == model.run_id
  assert restored_model.restored_from.epoch_num == 5

  assert restored_model.run_id != model.run_id

  # Check that mutated model state was restored.
  assert restored_model.epoch_num == 5
  assert restored_model.model_data == 15


def test_call(
  model: MockModel,
  epoch_type: epoch.Type,
  batch_iterator: batches.BatchIterator,
  logger: logging.Logger,
):
  """Test that call returns results."""
  model.Initialize()
  results = model(
    epoch_type=epoch_type, batch_iterator=batch_iterator, logger=logger
  )
  assert isinstance(results, epoch.Results)
  # Test that the model saw all of the input graphs.
  assert model.graph_count == batch_iterator.graph_count
  # Test that batch counts match up. More batches can be made than are used
  # (because the last batch could be empty).
  assert results.batch_count <= model.make_batch_count
  assert results.batch_count == model.run_batch_count
  # Check that result properties are propagated.
  # FIXME:
  # assert results.has_loss == model.has_loss
  # assert results.has_learning_rate == model.has_learning_rate


def test_empty_batches_is_not_error(
  graph_db: graph_tuple_database.Database, logger: logging.Logger,
):
  """Test that empty batches are ignored.

  Regression test for <github.com/ChrisCummins/ProGraML/issues/43>. Empty batch
  generation was determined to be the cause of flaky model crashes.
  """

  class FlakyBatchModel(MockModel):
    """A mock model which returns ~50% empty batches."""

    def __init__(self, *args, **kwargs):
      super(FlakyBatchModel, self).__init__(*args, **kwargs)
      self._batch_count = 0

    def MakeBatch(
      self,
      epoch_type: epoch.Type,
      graphs: Iterable[graph_tuple_database.GraphTuple],
      ctx: progress.ProgressContext = progress.NullContext,
    ) -> batches.Data:
      self._batch_count += 1

      if self._batch_count == 1:
        # Always return an empty first batch.
        return batches.EmptyBatch()
      elif self._batch_count == 2:
        # Always return a real second batch (otherwise the epoch may end up with
        # nothing but empty batches).
        return super(FlakyBatchModel, self).MakeBatch(epoch_type, graphs, ctx)

      # Return subsequent batches with 50% success rate.
      if random.random() < 0.5:
        return super(FlakyBatchModel, self).MakeBatch(epoch_type, graphs, ctx)
      else:
        return batches.EmptyBatch()

  model = FlakyBatchModel(logger=logger, graph_db=graph_db)

  batch_iterator = batches.BatchIterator(
    batches=model.BatchIterator(
      epoch.Type.TRAIN, graph_database_reader.BufferedGraphReader(graph_db)
    ),
    graph_count=graph_db.graph_count,
  )

  model.Initialize()
  results = model(
    epoch_type=epoch.Type.TRAIN, batch_iterator=batch_iterator, logger=logger
  )
  assert results.batch_count >= 1


if __name__ == "__main__":
  test.Main()
