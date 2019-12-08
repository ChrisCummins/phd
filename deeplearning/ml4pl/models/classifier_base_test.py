"""Unit tests for //deeplearning/ml4pl/models:classifier_base."""
import copy
import random
from typing import Iterable
from typing import Tuple

import numpy as np

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batches
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


@test.Fixture(scope="session", params=testing_databases.GetDatabaseUrls())
def log_db(request) -> log_database.Database:
  """A test fixture which yields an empty log database."""
  yield from testing_databases.YieldDatabase(
    log_database.Database, request.param
  )


@test.Fixture(scope="session")
def logger(log_db: log_database.Database) -> logging.Logger:
  """A test fixture which yields a logger."""
  with logging.Logger(
    log_db,
    max_buffer_size=None,
    max_buffer_length=128,
    max_seconds_since_flush=None,
  ) as logger:
    yield logger


@test.Fixture(scope="session", params=((0, 2), (0, 3), (2, 0), (10, 0)))
def y_dimensionalities(request) -> Tuple[int, int]:
  """A test fixture which enumerates node and graph label dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(0, 3))
def edge_position_max(request) -> int:
  """A test fixture which enumerates edge positions."""
  return request.param


@test.Fixture(scope="session", params=(None, "foo"))
def restored_from(request) -> str:
  """A test fixture for 'restored_from' values."""
  return request.param


@test.Fixture(scope="session", params=list(epoch.Type))
def epoch_type(request) -> epoch.Type:
  """A test fixture which enumerates epoch types."""
  return request.param


# Currently, only 2-dimension node features are supported.
@test.Fixture(scope="session", params=(2,))
def node_x_dimensionality(request) -> int:
  """A test fixture which enumerates node feature dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(0, 2))
def graph_x_dimensionality(request) -> int:
  """A test fixture which enumerates graph feature dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(True,))
def with_data_flow(request) -> int:
  """A test fixture which enumerates 'with dataflow' values."""
  return request.param


@test.Fixture(scope="session", params=(10, 100))
def graph_count(request) -> int:
  """A test fixture which enumerates graph counts."""
  return request.param


@test.Fixture(scope="session")
def graphs(
  graph_count: int,
  y_dimensionalities: Tuple[int, int],
  node_x_dimensionality: int,
  with_data_flow: bool,
) -> Iterable[graph_tuple_database.GraphTuple]:
  """A test fixture which enumerates graph tuples."""
  node_y_dimensionality, graph_y_dimensionality = y_dimensionalities

  def MakeIterator(list):
    yield from list

  return MakeIterator(
    [
      random_graph_tuple_database_generator.CreateRandomGraphTuple(
        node_x_dimensionality=node_x_dimensionality,
        node_y_dimensionality=node_y_dimensionality,
        graph_y_dimensionality=graph_y_dimensionality,
        with_data_flow=with_data_flow,
      )
      for _ in range(graph_count)
    ]
  )


class MockModel(classifier_base.ClassifierBase):
  """A mock classifier model."""

  def __init__(
    self, *args, has_loss: bool = True, has_learning_rate: bool = True, **kwargs
  ):
    super(MockModel, self).__init__(*args, **kwargs)
    # The "state" of the model.
    self.model_data = {"foo": 1, "last_graph_id": None}

    self.has_loss = has_loss
    self.has_learning_rate = has_learning_rate

    # Counters for testing method calls.
    self.make_batch_count = 0
    self.run_batch_count = 0
    self.graph_count = 0

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    self.make_batch_count += 1
    graph_ids = []
    while len(graph_ids) < 100:
      try:
        graph_ids.append(next(graphs).id)
      except StopIteration:
        break
    return batches.Data(graph_ids=graph_ids, data={"foo": 1})

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batches.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Results:
    # Update the counters.
    self.run_batch_count += 1
    self.graph_count += len(batch.graph_ids)
    self.model_data["last_graph_id"] = batch.graph_ids[-1]

    learning_rate = random.random() if self.has_learning_rate else None
    loss = random.random() if self.has_loss else None
    return batches.Results.Create(
      targets=np.random.rand(batch.graph_count, self.y_dimensionality),
      predictions=np.random.rand(batch.graph_count, self.y_dimensionality),
      iteration_count=random.randint(1, 3),
      model_converged=random.choice([False, True]),
      learning_rate=learning_rate,
      loss=loss,
    )

  def GetModelData(self):
    """Prepare data to save."""
    return self.model_data

  def LoadModelData(self, data_to_load):
    """Reset model state from loaded data."""
    self.model_data = copy.deepcopy(data_to_load)


@test.Fixture(scope="session", params=(False, True))
def has_loss(request) -> bool:
  """A test fixture which enumerates losses."""
  return request.param


@test.Fixture(scope="session", params=(False, True))
def has_learning_rate(request) -> bool:
  """A test fixture which enumerates learning rates."""
  return request.param


@test.Fixture(scope="function")
def model(
  node_x_dimensionality: int,
  graph_x_dimensionality: int,
  y_dimensionalities: Tuple[int, int],
  has_loss: bool,
  has_learning_rate: bool,
  edge_position_max: int,
  restored_from: str,
) -> MockModel:
  """A test fixture which enumerates mock models."""
  node_y_dimensionality, graph_y_dimensionality = y_dimensionalities
  run_id = run_id_lib.RunId.GenerateUnique(
    f"mock{random.randint(0, int(1e6)):06}"
  )

  return MockModel(
    run_id=run_id,
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    edge_position_max=edge_position_max,
    restored_from=restored_from,
    has_loss=has_loss,
    has_learning_rate=has_learning_rate,
  )


@test.Fixture(scope="function")
def batch_iterator(
  model: MockModel,
  graph_count: int,
  graphs: Iterable[graph_tuple_database.GraphTuple],
) -> batches.BatchIterator:
  return batches.BatchIterator(
    batches=model.BatchIterator(graphs), graph_count=graph_count
  )


def test_call_before_initialize(
  model: MockModel,
  epoch_type: epoch.Type,
  batch_iterator: batches.BatchIterator,
  logger: logging.Logger,
):
  """Test that call before initialize raises a TypeError."""
  with test.Raises(TypeError):
    model(epoch_type=epoch_type, batch_iterator=batch_iterator, logger=logger)


def test_call_returns_results(
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
  # Test that the model saw all graphs.
  model.graph_count == batch_iterator.graph_count
  # Test that batch counts match up.
  results.batch_count == model.make_batch_count == model.run_batch_count
  # Check that result properties are propagated.
  # FIXME:
  # assert results.has_loss == model.has_loss
  # assert results.has_learning_rate == model.has_learning_rate


@test.Fixture(scope="session", params=(0, 5))
def epoch_num(request) -> int:
  return request.param


###############################################################################
# Tests.
###############################################################################


def test_FromCheckpoint_direct(model: MockModel, epoch_num: int):
  """Test restoring directly from a model checkpoint."""
  model.Initialize()

  # Mutate model state.
  model.epoch_num = epoch_num
  model.model_data["some_new_data"] = 10

  # Create a new model from this checkpoint
  restored_model = MockModel.FromCheckpoint(
    model.GetCheckpoint(),
    node_y_dimensionality=model.node_y_dimensionality,
    graph_y_dimensionality=model.graph_y_dimensionality,
    edge_position_max=model.edge_position_max,
  )

  assert restored_model.restored_from == model.run_id

  # Check that mutated model state was restored.
  assert restored_model.epoch_num == epoch_num
  assert "some_new_data" in restored_model.model_data
  assert restored_model.model_data == model.model_data


def test_FromCheckpoint_via_logger(
  model: MockModel, epoch_num: int, logger: logging.Logger
):
  """Test restoring from a logged checkpoint."""
  model.Initialize()

  # Mutate model state.
  model.epoch_num = epoch_num
  model.model_data["some_new_data"] = 10

  # Create a new model from this checkpoint
  logger.Save(model.GetCheckpoint())

  restored_model = MockModel.FromCheckpoint(
    logger.Load(model.run_id, model.epoch_num)
  )

  assert restored_model.restored_from == model.run_id

  # Check that mutated model state was restored.
  assert restored_model.epoch_num == epoch_num
  assert "some_new_data" in restored_model.model_data
  assert restored_model.model_data == model.model_data


#   """Test loading a model from file."""
#
#
#   model = MockModel(graph_db, log_db)
#   model.InitializeModel()
#   model.epoch_num = 2
#   model.global_training_step = 10
#   model.mock_data = 100
#   model.SaveModel(validation_accuracy=0.5)
#
#   model.epoch_num = 0
#   model.mock_data = 0
#   model.global_training_step = 0
#   model.LoadModel(run_id=model.run_id, epoch_num=2)
#   assert model.epoch_num == 2
#   assert model.global_training_step == 10
#   assert model.mock_data == 100
#
#
# def test_LoadModel_unknown_saved_model_flag(
#   tempdir2: pathlib.Path,
#   graph_db: graph_database.Database,
#   log_db: log_database.Database,
# ):
#   """Test that error is raised if saved model contains unknown flag."""
#   FLAGS.working_dir = tempdir2
#   model = MockModel(graph_db, log_db)
#   model.InitializeModel()
#   model.SaveModel(validation_accuracy=0.5)
#
#   with model.log_db.Session(commit=True) as session:
#     # Insert a new "unknown" model flag.
#     session.add(
#       log_database.Parameter(
#         run_id=model.run_id,
#         type=log_database.ParameterType.MODEL_FLAG,
#         parameter="a new flag",
#         pickled_value=pickle.dumps(5),
#       )
#     )
#
#   with test.Raises(EnvironmentError) as e_ctx:
#     model.LoadModel(run_id=model.run_id, epoch_num=model.epoch_num)
#
#   # Check that the LoadModel() specifically complains about the new flag value.
#   assert "a new flag" in str(e_ctx.value)
#
#
# def test_ModelFlagsToDict_subclass_model_name(
#   tempdir2: pathlib.Path,
#   graph_db: graph_database.Database,
#   log_db: log_database.Database,
# ):
#   """Test that model name uses subclass name, not the base class."""
#   FLAGS.working_dir = tempdir2
#
#   model = MockModel(graph_db, log_db)
#   assert "model" in model.ModelFlagsToDict()
#   assert model.ModelFlagsToDict()["model"] == "MockModel"
#
#
# def test_Train(
#   tempdir2: pathlib.Path,
#   graph_db: graph_database.Database,
#   log_db: log_database.Database,
# ):
#   """Test that training terminates and bumps the epoch number."""
#   FLAGS.working_dir = tempdir2
#
#   model = MockModel(graph_db, log_db)
#   model.InitializeModel()
#   model.Train(epoch_count=1)
#   assert model.best_epoch_num == 1
#
#
# def test_Train_epoch_num(
#   tempdir2: pathlib.Path,
#   graph_db: graph_database.Database,
#   log_db: log_database.Database,
# ):
#   """Test that epoch_num has expected value."""
#   FLAGS.working_dir = tempdir2
#
#   model = MockModel(graph_db, log_db)
#   model.InitializeModel()
#   assert model.epoch_num == 0
#   model.Train(epoch_count=1)
#   assert model.epoch_num == 1
#   model.Train(epoch_count=1)
#   assert model.epoch_num == 2
#   model.Train(epoch_count=2)
#   assert model.epoch_num == 4
#
#
# def test_Train_batch_log_count(
#   tempdir2: pathlib.Path,
#   graph_db: graph_database.Database,
#   log_db: log_database.Database,
# ):
#   """Test that training produces only batch logs for {val,test} runs."""
#   FLAGS.working_dir = tempdir2
#
#   model = MockModel(graph_db, log_db)
#   model.InitializeModel()
#   model.Train(epoch_count=1)
#   with log_db.Session() as session:
#     # 10 train + 10 val + 10 test logs
#     assert session.query(log_database.BatchLogMeta).count() == 30
#     # 10 val + 10 test logs
#     assert session.query(log_database.BatchLog).count() == 20
#
#     query = session.query(log_database.BatchLogMeta)
#     query = query.filter(log_database.BatchLogMeta.type == "test")
#     for batch_log_meta in query:
#       assert batch_log_meta.batch_log
#
#     query = session.query(log_database.BatchLogMeta)
#     query = query.filter(log_database.BatchLogMeta.type == "train")
#     for batch_log_meta in query:
#       assert not batch_log_meta.batch_log
#
#   model.Train(epoch_count=1)
#   with log_db.Session() as session:
#     # + 10 train + 10 train (no change to val acc so there's no new test logs)
#     assert session.query(log_database.BatchLogMeta).count() == 50
#
#
# def test_Train_keeps_a_single_checkpoint_and_set_of_batch_logs(
#   tempdir2: pathlib.Path,
#   graph_db: graph_database.Database,
#   log_db: log_database.Database,
# ):
#   """Check that only a single model checkpoint and set of detailed val logs
#   are kept."""
#   FLAGS.working_dir = tempdir2
#
#   model = MockModel(graph_db, log_db)
#   model.InitializeModel()
#   model.Train(epoch_count=1)
#
#   # Force the model to believe that it performed worse than it did so that when
#   # we next call Train() it bumps the "best" accuracy.
#   log_db.engine.execute(
#     sql.update(log_database.ModelCheckpointMeta).values(
#       {"validation_accuracy": -1,}
#     )
#   )
#   assert model.best_epoch_validation_accuracy == -1  # Sanity check
#
#   model.Train(epoch_count=1)
#   # assert model.best_epoch_validation_accuracy == 1  # Sanity check
#
#   with log_db.Session() as session:
#     # There should still only be a single model checkpoint.
#     assert session.query(log_database.ModelCheckpoint).count() == 1
#     assert session.query(log_database.ModelCheckpointMeta).count() == 1
#
#     # The "best" epoch is the new one.
#     assert session.query(log_database.ModelCheckpointMeta.epoch).one()[0] == 2
#
#     # 10 val + 10 test
#     assert session.query(log_database.BatchLog).count() == 20
#     # Check that the new batch logs replace the old ones.
#     detailed_logs = session.query(log_database.BatchLogMeta)
#     detailed_logs = detailed_logs.join(log_database.BatchLog)
#     for log in detailed_logs:
#       log.epoch == 2
#
#
# def test_Test_creates_batch_logs(
#   tempdir2: pathlib.Path,
#   graph_db: graph_database.Database,
#   log_db: log_database.Database,
# ):
#   """Test that testing produces batch logs."""
#   FLAGS.working_dir = tempdir2
#
#   model = MockModel(graph_db, log_db)
#   model.InitializeModel()
#   model.RunEpoch(epoch_type="test")
#   with log_db.Session() as session:
#     assert session.query(log_database.BatchLogMeta).count() == 10
#     assert session.query(log_database.BatchLog).count() == 10
#
#
# def _MakeNRandomGraphs(n: int, group: str) -> typing.Iterable[nx.MultiDiGraph]:
#   """Private helper to generate random graphs of the given group."""
#   for i in range(n):
#     g = random_cdfg_generator.FastCreateRandom()
#     g.bytecode_id = 0
#     g.relpath = str(i)
#     g.language = "c"
#     g.group = group
#     g.source_name = "rand"
#     yield g


if __name__ == "__main__":
  test.Main()
