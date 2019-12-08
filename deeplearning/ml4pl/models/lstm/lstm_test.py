"""Unit tests for //deeplearning/ml4pl/models/lstm."""
import random
import string
from typing import List
from typing import Set

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logging
from deeplearning.ml4pl.models.lstm import lstm
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test

FLAGS = test.FLAGS


###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(scope="session", params=testing_databases.GetDatabaseUrls())
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


@test.Fixture(scope="session", params=(10, 100))
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


@test.Fixture(scope="session")
def opencl_relpaths() -> List[str]:
  opencl_df = make_devmap_dataset.MakeGpuDataFrame(
    opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df,
    "amd_tahiti_7970",
  )
  return list(set(opencl_df.relpath.values))


def PopulateOpenClGraphs(
  db: graph_tuple_database.Database,
  relpaths: List[str],
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
):
  """Populate a database with """
  rows = []
  # Create random rows using OpenCL relpaths.
  for i, relpath in enumerate(relpaths):
    graph_tuple = random_graph_tuple_database_generator.CreateRandomGraphTuple(
      node_x_dimensionality=2,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
    )
    graph_tuple.ir_id = i + 1
    graph_tuple.id = len(relpaths) - i
    rows.append(graph_tuple)

  with db.Session(commit=True) as session:
    session.add_all(rows)


@test.Fixture(scope="session", params=testing_databases.GetDatabaseUrls())
def graph_y_db(
  request, opencl_relpaths: List[str], graph_y_dimensionality: int,
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    PopulateOpenClGraphs(
      db,
      opencl_relpaths,
      node_y_dimensionality=0,
      graph_x_dimensionality=2,
      graph_y_dimensionality=graph_y_dimensionality,
    )
    yield db


def CreateRandomString(min_length: int = 1, max_length: int = 1024) -> str:
  """Generate a random string."""
  return "".join(
    random.choice(string.ascii_lowercase)
    for _ in range(random.randint(min_length, max_length))
  )


@test.Fixture(scope="session", params=testing_databases.GetDatabaseUrls())
def ir_db(request, opencl_relpaths: List[str]) -> ir_database.Database:
  """A test fixture which yields an IR database with 256 OpenCL entries."""
  with testing_databases.DatabaseContext(
    ir_database.Database, request.param
  ) as db:
    rows = []
    # Create IRs using OpenCL relpaths.
    for i, relpath in enumerate(opencl_relpaths):
      ir = ir_database.IntermediateRepresentation.CreateFromText(
        source="pact17_opencl_devmap",
        relpath=relpath,
        source_language=ir_database.SourceLanguage.OPENCL,
        type=ir_database.IrType.LLVM_6_0,
        cflags="",
        text=CreateRandomString(),
      )
      ir.id = i + 1
      rows.append(ir)

    with db.Session(commit=True) as session:
      session.add_all(rows)

    yield db


###############################################################################
# Tests.
###############################################################################


@test.Parametrize(
  "model_class", (lstm.LstmGraphClassifier, lstm.LstmNodeClassifier)
)
def test_load_restore_model_from_checkpoint_smoke_test(
  logger: logging.Logger,
  graph_y_db: graph_tuple_database.Database,
  ir_db: ir_database.Database,
  model_class,
):
  """Test creating and restoring model from checkpoint."""
  model: lstm.LstmBase = model_class(logger, graph_y_db, ir_db=ir_db)
  model.Initialize()

  checkpoint_ref = model.SaveCheckpoint()

  model.RestoreFrom(checkpoint_ref)


# def test_call(
#     epoch_type: epoch.Type,
#     logger: logging.Logger,
# ):
#   """Test that call returns results."""
#
#   results = model(
#       epoch_type=epoch_type, batch_iterator=batch_iterator, logger=logger
#   )
#   assert isinstance(results, epoch.Results)
#   # Test that the model saw all of the input graphs.
#   assert model.graph_count == batch_iterator.graph_count
#   # Test that batch counts match up. More batches can be made than are used
#   # (because the last batch could be empty).
#   assert results.batch_count <= model.make_batch_count
#   assert results.batch_count == model.run_batch_count
#
#   # Check that result properties are propagated.
#   # FIXME:
#   # assert results.has_loss == model.has_loss
#   # assert results.has_learning_rate == model.has_learning_rate


if __name__ == "__main__":
  test.Main()
