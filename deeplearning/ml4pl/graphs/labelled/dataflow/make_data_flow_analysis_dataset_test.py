"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow:make_data_flow_analysis_dataset."""
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.dataflow import (
  make_data_flow_analysis_dataset,
)
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.testing import random_programl_generator
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


@test.Fixture(scope="session", params=(1, 3))
def n(request) -> int:
  """Enumerate the number of labelled graphs per input proto."""
  return request.param


@test.Fixture(scope="session", params=("in_order", "random"))
def order_by(request) -> str:
  """Enumerate the number of --order_by options."""
  return request.param


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("proto_db"),
)
def proto_db(request,) -> unlabelled_graph_database.Database.SessionType:
  """A test fixture which yields a proto database."""
  with testing_databases.DatabaseContext(
    unlabelled_graph_database.Database, request.param
  ) as db:
    random_unlabelled_graph_database_generator.PopulateDatabaseWithRealProtos(
      db
    )
    yield db


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("proto_db"),
)
def proto_db_10(request,) -> unlabelled_graph_database.Database.SessionType:
  """A test fixture which yields a database with 10 protos."""
  with testing_databases.DatabaseContext(
    unlabelled_graph_database.Database, request.param
  ) as db:
    with db.Session(commit=True) as session:
      session.add_all(
        [
          unlabelled_graph_database.ProgramGraph.Create(proto, ir_id=i + 1)
          for i, proto in enumerate(
            list(random_programl_generator.EnumerateProtoTestSet())[:10]
          )
        ]
      )
    yield db


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def graph_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields an empty graph database."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


###############################################################################
# Tests.
###############################################################################


def test_pass_thru_analysis(
  proto_db: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  order_by: str,
  n: int,
):
  """Test that pass-thru annotator produces n * protos graphs."""
  FLAGS.n = n
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      input_db=proto_db,
      analysis="test_pass_thru",
      output_db=graph_db,
      order_by=order_by,
    )
  )
  with graph_db.Session() as session, proto_db.Session() as proto_session:
    # Check that n * proto_countto graphs were generated.
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      == n
      * proto_session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
      ).scalar()
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
  order_by: str,
  n: int,
):
  """Test that error annotator produces one 'empty' graph for each input."""
  FLAGS.n = n
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      input_db=proto_db,
      analysis="test_error",
      output_db=graph_db,
      order_by=order_by,
    )
  )
  with graph_db.Session() as session, proto_db.Session() as proto_session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      == proto_session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
      ).scalar()
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
  order_by: str,
  n: int,
):
  """Test that flaky annotator produces "some" graphs."""
  FLAGS.n = n
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      input_db=proto_db,
      analysis="test_flaky",
      output_db=graph_db,
      order_by=order_by,
    )
  )
  with graph_db.Session() as session, proto_db.Session() as proto_session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      >= proto_session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
      ).scalar()
    )

    # Not all graphs are empty.
    assert session.query(
      sql.func.sum(graph_tuple_database.GraphTuple.node_count)
    ).scalar()


def test_timeout_analysis(
  proto_db_10: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  order_by: str,
  n: int,
):
  """Test that timeout annotator produces one 'empty' graph for each input."""
  FLAGS.n = n
  FLAGS.annotator_timeout = 1
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      input_db=proto_db_10,
      analysis="test_timeout",
      output_db=graph_db,
      order_by=order_by,
    )
  )
  with graph_db.Session() as session, proto_db_10.Session() as proto_session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      == proto_session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
      ).scalar()
    )

    # All graphs are empty.
    assert (
      session.query(
        sql.func.sum(graph_tuple_database.GraphTuple.node_count)
      ).scalar()
      == 0
    )


def test_empty_analysis(
  proto_db: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  order_by: str,
  n: int,
):
  """Test that 'empty' graphs are produced when analysis returns no results."""
  FLAGS.n = n
  progress.Run(
    make_data_flow_analysis_dataset.DatasetGenerator(
      input_db=proto_db,
      analysis="test_empty",
      output_db=graph_db,
      order_by=order_by,
    )
  )
  with graph_db.Session() as session, proto_db.Session() as proto_session:
    output_graph_count = session.query(
      sql.func.count(graph_tuple_database.GraphTuple.id)
    ).scalar()

    input_graph_count = proto_session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
    ).scalar()

    assert output_graph_count == input_graph_count

    # All graphs are empty.
    assert (
      session.query(
        sql.func.sum(graph_tuple_database.GraphTuple.node_count)
      ).scalar()
      == 0
    )


if __name__ == "__main__":
  test.Main()
