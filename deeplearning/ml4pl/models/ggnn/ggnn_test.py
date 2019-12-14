"""Unit tests for //deeplearning/ml4pl/models/ggnn."""
import math
import random
from typing import Iterable

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import batch_iterator as batch_iterator_lib
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logging
from deeplearning.ml4pl.models.ggnn import ggnn
from deeplearning.ml4pl.models.ggnn import ggnn_config
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test
from labm8.py.internal import flags_parsers

FLAGS = test.FLAGS

# For testing models, always use --strict_graph_segmentation.
FLAGS.strict_graph_segmentation = True


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


@test.Fixture(scope="session", params=(50,))
def graph_count(request) -> int:
  """A test fixture which enumerates graph counts."""
  return request.param


@test.Fixture(scope="session", params=(0, 2))
def graph_x_dimensionality(request) -> int:
  """A test fixture which enumerates graph feature dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(2, 104))
def graph_y_dimensionality(request) -> int:
  """A test fixture which enumerates graph label dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(2, 3, 104))
def node_y_dimensionality(request) -> int:
  """A test fixture which enumerates graph label dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=list(epoch.Type))
def epoch_type(request) -> epoch.Type:
  """A test fixture which enumerates epoch types."""
  return request.param


@test.Fixture(scope="session", params=(False, True))
def log1p_graph_x(request) -> bool:
  """Enumerate --log1p_graph_x values."""
  return request.param


@test.Fixture(scope="session", params=list(ggnn_config.NodeTextEmbeddingType))
def node_text_embedding_type(request):
  return flags_parsers.EnumFlag(
    ggnn_config.NodeTextEmbeddingType, request.param
  )


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("node_y_db"),
)
def node_y_graph_db(
  request, graph_count: int, node_y_dimensionality: int,
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count,
      node_y_dimensionality=node_y_dimensionality,
      node_x_dimensionality=2,
      graph_y_dimensionality=0,
      split_count=3,
    )
    yield db


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_y_db"),
)
def graph_y_graph_db(
  request, graph_count: int, graph_y_dimensionality: int,
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count,
      node_x_dimensionality=2,
      node_y_dimensionality=0,
      graph_x_dimensionality=2,
      graph_y_dimensionality=graph_y_dimensionality,
      split_count=3,
    )
    yield db


###############################################################################
# Tests.
###############################################################################


def test_load_restore_model_from_checkpoint_smoke_test(
  logger: logging.Logger, node_y_graph_db: graph_tuple_database.Database,
):
  """Test creating and restoring model from checkpoint."""
  run_id = run_id_lib.RunId.GenerateUnique(
    f"mock{random.randint(0, int(1e6)):06}"
  )

  # Create and initialize an untrained model.
  model = ggnn.Ggnn(logger, node_y_graph_db, run_id=run_id)
  model.Initialize()

  # Smoke test save and restore.
  checkpoint_ref = model.SaveCheckpoint()
  model.RestoreFrom(checkpoint_ref)


def CheckResultsProperties(
  results: epoch.Results,
  graph_db: graph_tuple_database.Database,
  epoch_type: epoch.Type,
):
  """Check for various properties of well-formed epoch results."""
  assert isinstance(results, epoch.Results)
  # Check that the epoch contains batches.
  assert results.batch_count

  # Check that results contain a loss value.
  assert results.has_loss
  assert results.loss
  assert not math.isnan(results.loss)

  # Test that the epoch included every graph (of the appropriate type) in the
  # database.
  assert results.graph_count == graph_db.split_counts[epoch_type.value]


def test_node_classifier_call(
  epoch_type: epoch.Type,
  logger: logging.Logger,
  node_y_graph_db: graph_tuple_database.Database,
  node_text_embedding_type,
):
  """Test running a node classifier."""
  FLAGS.inst2vec_embeddings = node_text_embedding_type

  run_id = run_id_lib.RunId.GenerateUnique(
    f"mock{random.randint(0, int(1e6)):06}"
  )

  # Create and initialize an untrained model.
  model = ggnn.Ggnn(logger, node_y_graph_db, run_id=run_id)
  model.Initialize()

  # Run the model over some random graphs.
  batch_iterator = batch_iterator_lib.MakeBatchIterator(
    model=model,
    graph_db=node_y_graph_db,
    splits={epoch.Type.TRAIN: [0], epoch.Type.VAL: [1], epoch.Type.TEST: [2],},
    epoch_type=epoch_type,
  )

  results = model(
    epoch_type=epoch_type, batch_iterator=batch_iterator, logger=logger,
  )

  CheckResultsProperties(results, node_y_graph_db, epoch_type)


def test_graph_classifier_call(
  epoch_type: epoch.Type,
  logger: logging.Logger,
  graph_y_graph_db: graph_tuple_database.Database,
  node_text_embedding_type,
  log1p_graph_x: bool,
):
  """Test running a graph classifier."""
  FLAGS.inst2vec_embeddings = node_text_embedding_type
  FLAGS.log1p_graph_x = log1p_graph_x

  run_id = run_id_lib.RunId.GenerateUnique(
    f"mock{random.randint(0, int(1e6)):06}"
  )

  # Create and initialize an untrained model.
  model = ggnn.Ggnn(logger, graph_y_graph_db, run_id=run_id)
  model.Initialize()

  # Run the model over some random graphs.
  batch_iterator = batch_iterator_lib.MakeBatchIterator(
    model=model,
    graph_db=graph_y_graph_db,
    splits={epoch.Type.TRAIN: [0], epoch.Type.VAL: [1], epoch.Type.TEST: [2],},
    epoch_type=epoch_type,
  )

  results = model(
    epoch_type=epoch_type, batch_iterator=batch_iterator, logger=logger,
  )

  CheckResultsProperties(results, graph_y_graph_db, epoch_type)


if __name__ == "__main__":
  test.Main()
