"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database."""
import random

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import app
from labm8.py import decorators
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function", params=testing_databases.TEST_DB_URLS)
def db(request) -> unlabelled_graph_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    unlabelled_graph_database.Database, request.param
  )


def CreateRandomProto() -> programl_pb2.ProgramGraph:
  """Generate a random graph proto."""
  g = random_cdfg_generator.FastCreateRandom()
  return networkx_to_protos.NetworkXGraphToProgramGraphProto(g)


@test.Fixture(scope="function")
def two_graph_db_session(
  db: unlabelled_graph_database.Database,
) -> unlabelled_graph_database.Database.SessionType:
  a = unlabelled_graph_database.ProgramGraph.Create(
    proto=CreateRandomProto(), ir_id=0
  )
  b = unlabelled_graph_database.ProgramGraph.Create(
    proto=CreateRandomProto(), ir_id=1
  )

  with db.Session() as session:
    session.add_all([a, b])
    session.commit()

    # Sanity check that the graphs have been added to the database.
    assert (
      session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.id)
      ).scalar()
      == 2
    )
    assert (
      session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraphData.id)
      ).scalar()
      == 2
    )

    yield session


# Cascaded delete tests.


def test_cascaded_delete_from_session(
  two_graph_db_session: unlabelled_graph_database.Database.SessionType,
):
  """Test that cascaded delete works when deleting an object from the session."""
  session = two_graph_db_session

  # Delete the first graph.
  a = (
    session.query(unlabelled_graph_database.ProgramGraph)
    .filter(unlabelled_graph_database.ProgramGraph.ir_id == 0)
    .one()
  )
  session.delete(a)
  session.commit()

  # Check that only the one program remains.
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraph.id)
    ).scalar()
    == 1
  )
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraphData.id)
    ).scalar()
    == 1
  )
  assert (
    session.query(unlabelled_graph_database.ProgramGraph.ir_id).scalar() == 1
  )


def test_cascaded_delete_using_query(
  two_graph_db_session: unlabelled_graph_database.Database.SessionType,
):
  """Test that cascaded delete works when deleting results of query."""
  session = two_graph_db_session

  # Delete the first graph. Don't synchronize the session as we don't care
  # about the mapped objects.
  session.query(unlabelled_graph_database.ProgramGraph).filter(
    unlabelled_graph_database.ProgramGraph.ir_id == 0
  ).delete()
  session.commit()

  # Check that only the one program remains.
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraph.id)
    ).scalar()
    == 1
  )
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraphData.id)
    ).scalar()
    == 1
  )
  assert (
    session.query(unlabelled_graph_database.ProgramGraph.ir_id).scalar() == 1
  )


@decorators.loop_for(seconds=10)
def test_fuzz_ProgramGraph_Create(db: unlabelled_graph_database.Database):
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  with db.Session(commit=True) as session:
    session.add(
      unlabelled_graph_database.ProgramGraph.Create(
        proto=CreateRandomProto(), ir_id=random.randint(0, int(4e6))
      )
    )


if __name__ == "__main__":
  test.Main()
