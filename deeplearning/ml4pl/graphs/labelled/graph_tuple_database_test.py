"""Unit tests for //deeplearning/ml4pl/graphs/labelled:graph_tuple_database."""
import random

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function", params=testing_databases.TEST_DB_URLS)
def db(request) -> graph_tuple_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


def CreateRandomGraphTuple() -> graph_tuple.GraphTuple:
  """Generate a random graph tuple."""
  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)
  g = programl.ProgramGraphToNetworkX(proto)
  return graph_tuple.GraphTuple.CreateFromNetworkX(g)


@test.Fixture(scope="function")
def two_graph_db_session(
  db: graph_tuple_database.Database,
) -> graph_tuple_database.Database.SessionType:
  """A test fixture which yields a database with two graph tuples."""
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
    graph_tuple=CreateRandomGraphTuple(), ir_id=1
  )
  b = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
    graph_tuple=CreateRandomGraphTuple(), ir_id=2
  )

  with db.Session() as session:
    session.add_all([a, b])
    session.commit()

    # Sanity check that the graphs have been added to the database.
    assert (
      session.query(
        sql.func.count(graph_tuple_database.GraphTuple.ir_id)
      ).scalar()
      == 2
    )
    assert (
      session.query(
        sql.func.count(graph_tuple_database.GraphTupleData.id)
      ).scalar()
      == 2
    )

    yield session


# Cascaded delete tests.


def test_cascaded_delete_from_session(
  two_graph_db_session: graph_tuple_database.Database.SessionType,
):
  """Test that cascaded delete works when deleting an object from the session."""
  session = two_graph_db_session

  # Delete the first graph.
  a = (
    session.query(graph_tuple_database.GraphTuple)
    .filter(graph_tuple_database.GraphTuple.ir_id == 1)
    .one()
  )
  session.delete(a)
  session.commit()

  # Check that only the one program remains.
  assert (
    session.query(
      sql.func.count(graph_tuple_database.GraphTuple.ir_id)
    ).scalar()
    == 1
  )
  assert (
    session.query(
      sql.func.count(graph_tuple_database.GraphTupleData.id)
    ).scalar()
    == 1
  )
  assert session.query(graph_tuple_database.GraphTuple.ir_id).scalar() == 2


def test_cascaded_delete_using_query(
  two_graph_db_session: graph_tuple_database.Database.SessionType,
):
  """Test that cascaded delete works when deleting results of query."""
  session = two_graph_db_session

  # Delete the first graph. Don't synchronize the session as we don't care
  # about the mapped objects.
  session.query(graph_tuple_database.GraphTuple).filter(
    graph_tuple_database.GraphTuple.ir_id == 1
  ).delete()
  session.commit()

  # Check that only the one program remains.
  assert (
    session.query(
      sql.func.count(graph_tuple_database.GraphTuple.ir_id)
    ).scalar()
    == 1
  )
  assert (
    session.query(
      sql.func.count(graph_tuple_database.GraphTupleData.id)
    ).scalar()
    == 1
  )
  assert session.query(graph_tuple_database.GraphTuple.ir_id).scalar() == 2


@decorators.loop_for(seconds=10)
def test_fuzz_GraphTuple_Create(db: graph_tuple_database.Database):
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  with db.Session(commit=True) as session:
    session.add(
      graph_tuple_database.GraphTuple.CreateFromGraphTuple(
        graph_tuple=CreateRandomGraphTuple(), ir_id=random.randint(0, int(4e6))
      )
    )


if __name__ == "__main__":
  test.Main()
