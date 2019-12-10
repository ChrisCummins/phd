"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow:make_data_flow_analysis_dataset."""
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.dataflow import (
  make_data_flow_analysis_dataset,
)
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.testing import (
  random_unlabelled_graph_database_generator,
)
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import app
from labm8.py import progress
from labm8.py import test

FLAGS = app.FLAGS


###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def graph_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields an empty graph database."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


@test.Fixture(scope="session", params=(25, 100))
def proto_count(request) -> int:
  return request.param


@test.Fixture(scope="session", params=(1, 3))
def n(request) -> int:
  return request.param


@test.Fixture(scope="session", params=testing_databases.GetDatabaseUrls())
def proto_db(
  request, proto_count: int
) -> unlabelled_graph_database.Database.SessionType:
  """A test fixture which yields a proto database."""
  with testing_databases.DatabaseContext(
    unlabelled_graph_database.Database, request.param
  ) as db:
    random_unlabelled_graph_database_generator.PopulateDatabaseWithRandomProgramGraphs(
      db, proto_count
    )
    yield db


###############################################################################
# Tests.
###############################################################################


def test_pass_thru_analysis(
  proto_db: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  proto_count: int,
  n: int,
):
  """Test that pass-thru annotator produces n * protos graphs."""
  FLAGS.n = n
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      proto_db, "test_pass_thru", graph_db
    )
  )
  with graph_db.Session() as session, proto_db.Session() as proto_session:
    # Check that n * proto_count graphs were generated.
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      == n * proto_count
    )

    # Check that every unique proto appears in the graph database.
    assert set(
      row.ir_id
      for row in session.query(graph_tuple_database.GraphTuple.ir_id).all()
    ) == set(
      row.ir_id
      for row in proto_session.query(
        unlabelled_graph_database.ProgramGraph.ir_id
      )
    )

    # Check the node counts of the generated graphs.
    assert (
      session.query(
        sql.func.sum(graph_tuple_database.GraphTuple.node_count)
      ).scalar()
      == n
      * proto_session.query(
        sql.func.sum(unlabelled_graph_database.ProgramGraph.node_count)
      ).scalar()
    )


def test_error_analysis(
  proto_db: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  proto_count: int,
  n: int,
):
  """Test that error annotator produces one 'empty' graph for each input."""
  FLAGS.n = n
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      proto_db, "test_error", graph_db
    )
  )
  with graph_db.Session() as session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      == proto_count
    )

    # All graphs are empty.
    assert (
      session.query(
        sql.func.sum(graph_tuple_database.GraphTuple.node_count)
      ).scalar()
      == 0
    )


@test.Flaky(reason="Annotator is stochastic")
def test_flaky_analysis(
  proto_db: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  proto_count: int,
  n: int,
):
  """Test that flaky annotator produces "some" graphs."""
  FLAGS.n = n
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      proto_db, "test_flaky", graph_db
    )
  )
  with graph_db.Session() as session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      >= proto_count
    )

    # Not all graphs are empty.
    assert session.query(
      sql.func.sum(graph_tuple_database.GraphTuple.node_count)
    ).scalar()


def test_timeout_analysis(
  proto_db: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  proto_count: int,
  n: int,
):
  """Test that timeout annotator produces one 'empty' graph for each input."""
  FLAGS.n = n
  FLAGS.annotator_timeout = 2
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      proto_db, "test_timeout", graph_db
    )
  )
  with graph_db.Session() as session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      == proto_count
    )

    # All graphs are empty.
    assert (
      session.query(
        sql.func.sum(graph_tuple_database.GraphTuple.node_count)
      ).scalar()
      == 0
    )


# TODO: Test timeout.
# TODO: Test error.


if __name__ == "__main__":
  test.Main()
