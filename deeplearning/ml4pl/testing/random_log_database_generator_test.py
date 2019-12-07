"""Unit tests for //deeplearning/ml4pl/testing:random_log_database_generator."""
import sqlalchemy as sql

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.testing import random_log_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="session")
def generator() -> random_log_database_generator.RandomLogDatabaseGenerator:
  """A test fixture which returns a log generator."""
  return random_log_database_generator.RandomLogDatabaseGenerator()


@test.Fixture(scope="session")
def run_id() -> run_id_lib.RunId:
  return run_id_lib.RunId.GenerateUnique("test")


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def db(request) -> log_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    log_database.Database, request.param
  )


@test.Fixture(scope="function")
def db_session(db: log_database.Database) -> log_database.Database.SessionType:
  """A test fixture which yields an empty graph proto database."""
  with db.Session() as session:
    yield session


@test.Parametrize("max_param_count", (1, 10))
@test.Parametrize(
  "graph_db", (None, graph_tuple_database.Database("sqlite://"))
)
def test_parameters(
  generator: random_log_database_generator.RandomLogDatabaseGenerator,
  run_id: run_id_lib.RunId,
  max_param_count: int,
  db_session: log_database.Database.SessionType,
  graph_db: graph_tuple_database.Database,
):
  """Black-box test of generator properties."""
  logs = generator.CreateRandomRunLogs(
    run_id=run_id, max_param_count=max_param_count, graph_db=graph_db
  )
  assert 1 <= len(logs.parameters) <= max_param_count
  for param in logs.parameters:
    assert isinstance(param, log_database.Parameter)
    assert param.run_id == run_id

  db_session.add_all(logs.all)
  db_session.commit()


@test.Parametrize("max_epoch_count", (1, 10))
@test.Parametrize("max_batch_count", (1, 10))
def test_batches(
  generator: random_log_database_generator.RandomLogDatabaseGenerator,
  run_id: run_id_lib.RunId,
  max_epoch_count: int,
  max_batch_count: int,
  db_session: log_database.Database.SessionType,
):
  """Black-box test of generator properties."""
  logs = generator.CreateRandomRunLogs(
    run_id=run_id,
    max_epoch_count=max_epoch_count,
    max_batch_count=max_batch_count,
  )
  assert 2 <= len(logs.batches) <= 3 * max_epoch_count * max_batch_count
  for batch in logs.batches:
    assert isinstance(batch, log_database.Batch)
    assert batch.run_id == run_id

  db_session.add_all(logs.all)
  db_session.commit()


@test.Fixture(scope="function", params=(1, 10))
def run_count(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(1, 10))
def max_param_count(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(1, 10))
def max_epoch_count(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(1, 10))
def max_batch_count(request) -> int:
  return request.param


def test_PopulateLogDatabase(
  generator: random_log_database_generator.RandomLogDatabaseGenerator,
  db: log_database.Database,
  run_count: int,
  max_param_count: int,
  max_epoch_count: int,
  max_batch_count: int,
):
  """Test populating databases."""
  generator.PopulateLogDatabase(
    db,
    run_count,
    max_param_count=max_param_count,
    max_epoch_count=max_epoch_count,
    max_batch_count=max_batch_count,
  )
  with db.Session() as session:
    assert (
      session.query(
        sql.func.count(sql.func.distinct(log_database.Parameter.run_id))
      ).scalar()
      == run_count
    )

    assert (
      session.query(
        sql.func.count(sql.func.distinct(log_database.Batch.run_id))
      ).scalar()
      == run_count
    )

    assert (
      session.query(
        sql.func.count(sql.func.distinct(log_database.Checkpoint.run_id))
      ).scalar()
      <= run_count
    )


def test_benchmark_CreateRandomGraphTuple(benchmark):
  """Benchmark logs generation."""

  def Benchmark():
    """A microbenchmark that instantiates a new generator."""
    generator = random_log_database_generator.RandomLogDatabaseGenerator()
    generator.CreateRandomRunLogs()

  benchmark(Benchmark)


def test_benchmark_CreateRandomGraphTuple_reuse_pool(
  benchmark, generator: random_log_database_generator.RandomLogDatabaseGenerator
):
  """Benchmark logs generation with re-used generator."""
  benchmark(generator.CreateRandomRunLogs)


if __name__ == "__main__":
  test.Main()
