# Copyright 2019-2020 the ProGraML authors.
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
"""Unit tests for //deeplearning/ml4pl/models:run."""
import copy
import random
from typing import Iterable
from typing import List

import numpy as np

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_analysis
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


@test.Fixture(scope="session")
def graph_db() -> graph_tuple_database.Database:
  """A test fixture which creates a session-level graph database."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, testing_databases.GetDatabaseUrls()[0]
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count=20,
      node_x_dimensionality=2,
      node_y_dimensionality=0,
      graph_x_dimensionality=2,
      graph_y_dimensionality=2,
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
    epoch_type: epoch.Type,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Generate a fake batch of data."""
    del epoch_type  # Unused.
    del ctx  # Unused.

    graph_ids = []
    while len(graph_ids) < 100:
      try:
        graph_ids.append(next(graphs).id)
      except StopIteration:
        if not graph_ids:
          return batches.EndOfBatches()
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


@test.Parametrize("k_fold", (False, True), names=["one_run", "k_fold"])
@test.Parametrize(
  "run_with_memory_profiler",
  (False, True),
  names=("no_memprof", "with_memprof"),
)
@test.Parametrize(
  "test_on",
  ("best", "none", "every", "improvement", "improvement_and_last"),
  namer=lambda x: f"test_on={x}",
)
@test.Parametrize(
  "stop_at",
  (["val_acc=.6"], ["time=200"], ["patience=5"]),
  namer=lambda x: f"stop_at={','.join(x)}",
)
def test_Run(
  disposable_log_db: log_database.Database,
  graph_db: graph_tuple_database.Database,
  k_fold: bool,
  run_with_memory_profiler: bool,
  test_on: str,
  stop_at: List[str],
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
  FLAGS.epoch_count = 3
  FLAGS.k_fold = k_fold
  FLAGS.run_with_memory_profiler = run_with_memory_profiler
  FLAGS.test_on = test_on
  FLAGS.stop_at = stop_at

  run.Run(MockModel)

  # Test that k-fold produces multiple runs.
  assert log_db.run_count == graph_db.split_count if k_fold else 1

  run_ids = log_db.run_ids
  for run_id in run_ids:
    logs = log_analysis.RunLogAnalyzer(log_db=log_db, run_id=run_id)
    epochs = logs.tables["epochs"]

    # Check that we performed as many epochs as expected. We can't check the
    # exact value because of --stop_at options.
    assert 1 <= len(epochs) <= FLAGS.epoch_count

    test_count = len(epochs[epochs["test_accuracy"].notnull()])

    # Test that the number of test epochs matches the expected amount depending
    # on --test_on flag.
    if test_on == "none":
      assert test_count == 0
    elif test_on == "best":
      assert test_count == 1
    elif test_on == "improvement":
      assert test_count >= 1
    elif test_on == "improvement_and_last":
      assert test_count >= 1


@test.Parametrize("k_fold", (False, True), names=["one_run", "k_fold"])
def test_Run_test_only(
  disposable_log_db: log_database.Database,
  graph_db: graph_tuple_database.Database,
  k_fold: bool,
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
  FLAGS.test_only = True
  FLAGS.k_fold = k_fold

  run.Run(MockModel)

  # Test that k-fold produces multiple runs.
  assert log_db.run_count == graph_db.split_count if k_fold else 1

  run_ids = log_db.run_ids
  for run_id in run_ids:
    logs = log_analysis.RunLogAnalyzer(log_db=log_db, run_id=run_id)
    epochs = logs.tables["epochs"]

    # Check that we performed as many epochs as expected.
    assert 1 == len(epochs)
    test_count = len(epochs[epochs["test_accuracy"].notnull()])
    # Check that we produced a test result.
    assert test_count == 1


if __name__ == "__main__":
  test.Main()
