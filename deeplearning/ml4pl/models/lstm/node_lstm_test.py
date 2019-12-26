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
"""Unit tests for //deeplearning/ml4pl/models/lstm:node_lstm."""
import random
from typing import List

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.models import batch_iterator as batch_iterator_lib
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logging
from deeplearning.ml4pl.models.lstm import node_lstm
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import (
  random_unlabelled_graph_database_generator,
)
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test

FLAGS = test.FLAGS

# For testing models, always use --strict_graph_segmentation.
FLAGS.strict_graph_segmentation = True

# For testing the LSTM we can use a reduced size model.
FLAGS.lang_model_hidden_size = 8
FLAGS.heuristic_model_hidden_size = 4

# No test coverage for hot code.
MODULE_UNDER_TEST = None

###############################################################################
# Fixtures.
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


@test.Fixture(scope="session", params=(2, 3), namer=lambda x: f"node_y:{x}")
def node_y_dimensionality(request) -> int:
  """A test fixture which enumerates graph label dimensionalities."""
  return request.param


@test.Fixture(
  scope="session", params=list(epoch.Type), namer=lambda x: x.name.lower()
)
def epoch_type(request) -> epoch.Type:
  """A test fixture which enumerates epoch types."""
  return request.param


@test.Fixture(scope="session")
def opencl_relpaths() -> List[str]:
  opencl_df = make_devmap_dataset.MakeGpuDataFrame(
    opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df,
    "amd_tahiti_7970",
  )
  return list(set(opencl_df.relpath.values))


@test.Fixture(scope="session")
def proto_db(opencl_relpaths: List[str]) -> unlabelled_graph_database.Database:
  """A test fixture which yields a proto database with 256 OpenCL entries."""
  with testing_databases.DatabaseContext(
    unlabelled_graph_database.Database, testing_databases.GetDatabaseUrls()[0]
  ) as db:
    random_unlabelled_graph_database_generator.PopulateDatabaseWithTestSet(
      db, len(opencl_relpaths)
    )

    yield db


@test.Fixture(scope="session")
def graph_db(
  opencl_relpaths: List[str], node_y_dimensionality: int,
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 100 real graphs."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, testing_databases.GetDatabaseUrls()[0]
  ) as db:
    random_graph_tuple_database_generator.PopulateWithTestSet(
      db,
      len(opencl_relpaths),
      node_x_dimensionality=2,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=0,
      graph_y_dimensionality=0,
      split_count=3,
    )
    yield db


###############################################################################
# Tests.
###############################################################################


def test_load_restore_model_from_checkpoint_smoke_test(
  logger: logging.Logger,
  graph_db: graph_tuple_database.Database,
  proto_db: unlabelled_graph_database.Database,
):
  """Test creating and restoring a model from checkpoint."""
  # Create and initialize a model.
  model = node_lstm.NodeLstm(
    logger,
    graph_db,
    proto_db=proto_db,
    batch_size=32,
    padded_sequence_length=10,
    padded_node_sequence_length=10,
  )
  model.Initialize()

  # Create a checkpoint from the model.
  checkpoint_ref = model.SaveCheckpoint()

  # Reset the model state to the checkpoint.
  model.RestoreFrom(checkpoint_ref)

  # Run a test epoch to make sure the restored model works.
  batch_iterator = batch_iterator_lib.MakeBatchIterator(
    model=model,
    graph_db=graph_db,
    splits={epoch.Type.TRAIN: [0], epoch.Type.VAL: [1], epoch.Type.TEST: [2],},
    epoch_type=epoch.Type.TEST,
  )
  model(
    epoch_type=epoch.Type.TEST, batch_iterator=batch_iterator, logger=logger,
  )

  # Create a new model instance and restore its state from the checkpoint.
  new_model = node_lstm.NodeLstm(
    logger,
    graph_db,
    proto_db=proto_db,
    batch_size=32,
    padded_sequence_length=10,
    padded_node_sequence_length=10,
  )
  new_model.RestoreFrom(checkpoint_ref)

  # Check that the new model works.
  batch_iterator = batch_iterator_lib.MakeBatchIterator(
    model=new_model,
    graph_db=graph_db,
    splits={epoch.Type.TRAIN: [0], epoch.Type.VAL: [1], epoch.Type.TEST: [2],},
    epoch_type=epoch.Type.TEST,
  )
  new_model(
    epoch_type=epoch.Type.TEST, batch_iterator=batch_iterator, logger=logger,
  )


def test_classifier_call(
  epoch_type: epoch.Type,
  logger: logging.Logger,
  graph_db: graph_tuple_database.Database,
  proto_db: unlabelled_graph_database.Database,
):
  """Test running a node classifier."""
  model = node_lstm.NodeLstm(
    logger,
    graph_db,
    proto_db=proto_db,
    batch_size=32,
    padded_sequence_length=100,
    padded_node_sequence_length=50,
  )
  model.Initialize()

  batch_iterator = batch_iterator_lib.MakeBatchIterator(
    model=model,
    graph_db=graph_db,
    splits={epoch.Type.TRAIN: [0], epoch.Type.VAL: [1], epoch.Type.TEST: [2],},
    epoch_type=epoch_type,
  )

  results = model(
    epoch_type=epoch_type, batch_iterator=batch_iterator, logger=logger,
  )
  assert isinstance(results, epoch.Results)

  assert results.batch_count

  # We only get loss for training.
  if epoch_type == epoch.Type.TRAIN:
    assert results.has_loss
  else:
    assert not results.has_loss


if __name__ == "__main__":
  test.Main()
