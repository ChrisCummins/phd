"""Unit tests for //deeplearning/ml4pl/models:log_database."""
from deeplearning.ml4pl import run_id
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.testing import random_log_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function", params=testing_databases.TEST_DB_URLS)
def db(request) -> log_database.Database:
  """A test fixture which yields an empty log database."""
  yield from testing_databases.YieldDatabase(
    log_database.Database, request.param
  )


@test.Fixture(scope="function")
def two_run_id_session(
  db: log_database.Database,
) -> log_database.Database.SessionType:
  """A test fixture which yields a database with two runs."""
  with db.Sesson() as session:
    run_id_a = run_id.RunId.GenerateUnique("a")
    run_id_b = run_id.RunId.GenerateUnique("b")
    a = random_log_database_generator.CreateRandomRunLogs(run_id=run_id_a)
    b = random_log_database_generator.CreateRandomRunLogs(run_id=run_id_b)
    session.add_all(a + b)
    session.commit()
    yield session


def test_BatchLogMeta_columns(db: log_database.Database):
  pass
  # with db.Session(commit=True) as session:
  #   session.add(MakeBatchLog())
  #
  # with db.Session() as session:
  #   log = session.query(log_database.BatchLogMeta).first()
  #   assert log.run_id == "20191023@foo"
  #   assert log.epoch == 10
  #   assert log.batch == 0
  #   assert log.global_step == 1024
  #   assert log.elapsed_time_seconds == 0.5
  #   assert log.graph_count == 100
  #   assert log.node_count == 500
  #   assert log.loss == 0.25
  #   assert log.precision == 0.5
  #   assert log.recall == 0.5
  #   assert log.f1 == 0.5
  #   assert log.accuracy == 0.75
  #   assert log.type == "train"
  #   assert log.group == "0"
  #   assert log.graph_indices == [0, 1, 2, 3]
  #   assert np.array_equal(log.predictions, np.array([0, 1, 2, 3]))
  #   assert np.array_equal(log.accuracies, np.array([True, False, False]))


def test_DeleteLogsForRunId():
  """Test that delete batch log meta cascades to batch log."""
  pass
  # with db.Session(commit=True) as session:
  #   log = MakeBatchLog()
  #   run_id = log.run_id
  #   session.add(log)
  #
  #   session.add(
  #     log_database.Parameter(
  #       run_id=run_id,
  #       parameter="foo",
  #       type=log_database.ParameterType.MODEL_FLAG,
  #       pickled_value=pickle.dumps("foo"),
  #     )
  #   )
  #
  # db.DeleteLogsForRunId(run_id)
  #
  # with db.Session() as session:
  #   assert not session.query(log_database.BatchLogMeta.id).count()
  #   assert not session.query(log_database.BatchLog.id).count()
  #   assert not session.query(log_database.Parameter.id).count()


def test_run_ids():
  """Test that property returns all run IDs."""
  pass
  # with db.Session(commit=True) as session:
  #   session.add_all(
  #     [
  #       log_database.Parameter(
  #         run_id="a",
  #         type=log_database.ParameterType.MODEL_FLAG,
  #         parameter="foo",
  #         pickled_value=pickle.dumps("foo"),
  #       ),
  #       log_database.Parameter(
  #         run_id="a",
  #         type=log_database.ParameterType.MODEL_FLAG,
  #         parameter="bar",
  #         pickled_value=pickle.dumps("bar"),
  #       ),
  #       log_database.Parameter(
  #         run_id="b",
  #         type=log_database.ParameterType.MODEL_FLAG,
  #         parameter="foo",
  #         pickled_value=pickle.dumps("foo"),
  #       ),
  #     ]
  #   )
  # assert db.run_ids == ["a", "b"]


if __name__ == "__main__":
  test.Main()
