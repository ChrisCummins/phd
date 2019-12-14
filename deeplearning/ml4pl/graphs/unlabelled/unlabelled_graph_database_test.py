"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database."""
import random

import sqlalchemy as sql

from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.testing import random_programl_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import app
from labm8.py import decorators
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def db(request) -> unlabelled_graph_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    unlabelled_graph_database.Database, request.param
  )


@test.Fixture(scope="function")
def two_graph_db_session(
  db: unlabelled_graph_database.Database,
) -> unlabelled_graph_database.Database.SessionType:
  a = unlabelled_graph_database.ProgramGraph.Create(
    proto=random_programl_generator.CreateRandomProto(), ir_id=1
  )
  b = unlabelled_graph_database.ProgramGraph.Create(
    proto=random_programl_generator.CreateRandomProto(), ir_id=2
  )

  with db.Session() as session:
    session.add_all([a, b])
    session.commit()

    # Sanity check that the graphs have been added to the database.
    assert (
      session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
      ).scalar()
      == 2
    )
    assert (
      session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraphData.ir_id)
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
    .filter(unlabelled_graph_database.ProgramGraph.ir_id == 1)
    .one()
  )
  session.delete(a)
  session.commit()

  # Check that only the one program remains.
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
    ).scalar()
    == 1
  )
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraphData.ir_id)
    ).scalar()
    == 1
  )
  assert (
    session.query(unlabelled_graph_database.ProgramGraph.ir_id).scalar() == 2
  )


def test_cascaded_delete_using_query(
  two_graph_db_session: unlabelled_graph_database.Database.SessionType,
):
  """Test that cascaded delete works when deleting results of query."""
  session = two_graph_db_session

  # Delete the first graph. Don't synchronize the session as we don't care
  # about the mapped objects.
  session.query(unlabelled_graph_database.ProgramGraph).filter(
    unlabelled_graph_database.ProgramGraph.ir_id == 1
  ).delete()
  session.commit()

  # Check that only the one program remains.
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
    ).scalar()
    == 1
  )
  assert (
    session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraphData.ir_id)
    ).scalar()
    == 1
  )
  assert (
    session.query(unlabelled_graph_database.ProgramGraph.ir_id).scalar() == 2
  )


# Database stats tests.

# Repeat test repeatedly to test memoized property accessor.
@decorators.loop_for(min_iteration_count=3)
def test_fuzz_database_stats_on_empty_db(
  db: unlabelled_graph_database.Database,
):
  assert db.proto_count == 0
  assert db.stats_json


# Global counter for test_fuzz_ProgramGraph_Create() to generate unique values.
ir_id = 0


@decorators.loop_for(seconds=30)
def test_fuzz_ProgramGraph_Create(db: unlabelled_graph_database.Database):
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  global ir_id
  ir_id += 1
  with db.Session(commit=True) as session:
    session.add(
      unlabelled_graph_database.ProgramGraph.Create(
        proto=random_programl_generator.CreateRandomProto(),
        ir_id=ir_id,
        split=random.randint(0, 10) if random.random() < 0.5 else None,
      )
    )


if __name__ == "__main__":
  test.Main()
