"""Unit tests for //deeplearning/ml4pl/models/lstm:graph_lstm."""
import random
import string
from typing import List

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.models import batch_iterator as batch_iterator_lib
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logging
from deeplearning.ml4pl.models.lstm import graph_lstm
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test

FLAGS = test.FLAGS

# For testing models, always use --strict_graph_segmentation.
FLAGS.strict_graph_segmentation = True


###############################################################################
# Utility functions.
###############################################################################


def CreateRandomString(min_length: int = 1, max_length: int = 1024) -> str:
  """Generate a random string."""
  return "".join(
    random.choice(string.ascii_lowercase)
    for _ in range(random.randint(min_length, max_length))
  )


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


@test.Fixture(scope="session", params=(2, 104), namer=lambda x: f"graph_y:{x}")
def graph_y_dimensionality(request) -> int:
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


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("ir_db"),
)
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


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def graph_db(
  request, opencl_relpaths: List[str], graph_y_dimensionality: int,
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateWithTestSet(
      db,
      len(opencl_relpaths),
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


@test.Parametrize("model_class", graph_lstm.GraphLstm)
@test.XFail(
  reason="TODO(github.com/ChrisCummins/ProGraML/issues/24): Cannot use the given session to evaluate tensor: the tensor's graph is different from the session's graph"
)
def test_load_restore_model_from_checkpoint_smoke_test(
  logger: logging.Logger,
  graph_db: graph_tuple_database.Database,
  ir_db: ir_database.Database,
):
  """Test creating and restoring model from checkpoint."""
  run_id = run_id_lib.RunId.GenerateUnique(
    f"mock{random.randint(0, int(1e6)):06}"
  )

  model = graph_lstm.GraphLstm(logger, graph_db, ir_db=ir_db, run_id=run_id)
  model.Initialize()

  checkpoint_ref = model.SaveCheckpoint()

  model.RestoreFrom(checkpoint_ref)


def test_classifier_call(
  epoch_type: epoch.Type,
  logger: logging.Logger,
  graph_db: graph_tuple_database.Database,
  ir_db: ir_database.Database,
):
  """Test running a graph classifier."""
  run_id = run_id_lib.RunId.GenerateUnique(
    f"mock{random.randint(0, int(1e6)):06}"
  )

  model = graph_lstm.GraphLstm(
    logger,
    graph_db,
    ir_db=ir_db,
    batch_size=8,
    padded_sequence_length=100,
    run_id=run_id,
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
