"""Unit tests for //deeplearning/ml4pl/graphs/labelled:graph_tuple_database."""
import random

import pytest
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import random_graph_tuple_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def db(request) -> graph_tuple_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


@test.Fixture(scope="function")
def two_graph_db_session(
  db: graph_tuple_database.Database,
) -> graph_tuple_database.Database.SessionType:
  """A test fixture which yields a database with two graph tuples."""
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
    graph_tuple=random_graph_tuple_generator.CreateRandomGraphTuple(), ir_id=1
  )
  b = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
    graph_tuple=random_graph_tuple_generator.CreateRandomGraphTuple(), ir_id=2
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


# CreateFromGraphTuple() tests.


def test_CreateFromGraphTuple_attributes():
  """Test that attributes are copied over."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple()
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert a.ir_id == 1
  assert a.node_count == graph_tuple.node_count
  assert a.edge_count == graph_tuple.edge_count
  assert a.control_edge_count == graph_tuple.control_edge_count
  assert a.data_edge_count == graph_tuple.data_edge_count
  assert a.call_edge_count == graph_tuple.call_edge_count

  assert a.edge_position_max == graph_tuple.edge_position_max


def test_CreateFromGraphTuple_node_x_dimensionality(
  db_session: graph_tuple_database.Database.SessionType,
):
  """Test node feature dimensionality."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple()
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert a.node_x_dimensionality == 1
  db_session.add(a)
  db_session.commit()


def test_CreateFromGraphTuple_node_y_dimensionality(
  db_session: graph_tuple_database.Database.SessionType,
):
  """Test node label dimensionality."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    node_y_dimensionality=0
  )
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert a.node_y_dimensionality == 0

  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    node_y_dimensionality=2
  )
  b = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert b.node_y_dimensionality == 2

  db_session.add_all([a, b])
  db_session.commit()


def test_CreateFromGraphTuple_graph_x_dimensionality(
  db_session: graph_tuple_database.Database.SessionType,
):
  """Check graph label dimensionality."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    graph_x_dimensionality=0
  )
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert a.graph_x_dimensionality == 0

  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    graph_x_dimensionality=2
  )
  b = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert b.graph_x_dimensionality == 2

  db_session.add_all([a, b])
  db_session.commit()


def test_CreateFromGraphTuple_graph_y_dimensionality(
  db_session: graph_tuple_database.Database.SessionType,
):
  """Check graph label dimensionality."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    graph_y_dimensionality=0
  )
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert a.graph_y_dimensionality == 0

  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    graph_y_dimensionality=2
  )
  b = graph_tuple_database.GraphTuple.CreateFromGraphTuple(graph_tuple, ir_id=1)
  assert b.graph_y_dimensionality == 2

  db_session.add_all([a, b])
  db_session.commit()


def test_CreateEmpty(db_session: graph_tuple_database.Database.SessionType):
  """Test creation of empty graph tuple."""
  a = graph_tuple_database.GraphTuple.CreateEmpty(ir_id=1)
  assert a.ir_id == 1

  db_session.add(a)
  db_session.commit()


@decorators.loop_for(seconds=10)
def test_fuzz_GraphTuple_Create(db: graph_tuple_database.Database):
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  with db.Session(commit=True) as session:
    graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple()
    t = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
      graph_tuple=graph_tuple, ir_id=random.randint(0, int(4e6))
    )
    assert t.edge_count == (
      t.control_edge_count + t.data_edge_count + t.call_edge_count
    )
    assert len(t.sha1) == 40
    assert t.node_count == graph_tuple.node_count
    assert t.edge_count == graph_tuple.edge_count
    assert t.tuple.node_count == graph_tuple.node_count
    assert t.tuple.edge_count == graph_tuple.edge_count
    assert len(t.tuple.adjacencies) == 3
    assert len(t.tuple.edge_positions) == 3
    session.add(t)


# Fixtures for enumerating populated databases.


@test.Fixture(scope="function", params=(1, 1000, 5000))
def graph_count(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(1, 3))
def node_x_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(0, 3))
def node_y_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(0, 3))
def graph_x_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(0, 3))
def graph_y_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="function", params=(False, True))
def with_data_flow(request) -> bool:
  return request.param


@test.Fixture(scope="function", params=(0, 2))
def split_count(request) -> int:
  return request.param


@test.Fixture(scope="function")
def populated_db_and_rows(
  db: graph_tuple_database.Database,
  graph_count: int,
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
  with_data_flow: bool,
  split_count: int,
) -> random_graph_tuple_database_generator.DatabaseAndRows:
  """Generate a populated database and a list of rows."""
  return random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
    db,
    graph_count,
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    with_data_flow=with_data_flow,
    split_count=split_count,
  )


# Database stats tests.

# Repeat test repeatedly to test memoized property accessor.
@decorators.loop_for(min_iteration_count=3)
def test_fuzz_database_stats_on_empty_db(db: graph_tuple_database.Database):
  assert db.graph_count == 0
  assert db.ir_count == 0


# Repeat test repeatedly to test memoized property accessor.
@decorators.loop_for(min_iteration_count=3)
def test_fuzz_database_stats(
  populated_db_and_rows: random_graph_tuple_database_generator.DatabaseAndRows,
):
  db, rows = populated_db_and_rows

  # Graph and IR counts.
  assert db.graph_count == len(rows)
  assert db.ir_count == len(set(r.ir_id for r in rows))
  assert db.split_count == split_count

  # Node and edge attributes.
  assert db.node_count == sum(r.node_count for r in rows)
  assert db.edge_count == sum(r.edge_count for r in rows)
  assert db.control_edge_count == sum(r.control_edge_count for r in rows)
  assert db.data_edge_count == sum(r.data_edge_count for r in rows)
  assert db.call_edge_count == sum(r.call_edge_count for r in rows)

  # Node and edge attribute maximums.
  assert db.node_count_max == max(r.node_count for r in rows)
  assert db.edge_count_max == max(r.edge_count for r in rows)
  assert db.control_edge_count_max == max(r.control_edge_count for r in rows)
  assert db.data_edge_count_max == max(r.data_edge_count for r in rows)
  assert db.call_edge_count_max == max(r.call_edge_count for r in rows)

  # Edge position max.
  assert db.edge_position_max == max(t.edge_position_max for t in rows)

  # Feature and label dimensionalities.
  assert db.node_x_dimensionality == rows[0].node_x_dimensionality
  assert db.node_y_dimensionality == rows[0].node_y_dimensionality
  assert db.graph_x_dimensionality == rows[0].graph_x_dimensionality
  assert db.graph_y_dimensionality == rows[0].graph_y_dimensionality

  assert db.graph_data_size == sum(r.pickled_graph_tuple_size for r in rows)
  assert db.graph_data_size_min == min(r.pickled_graph_tuple_size for r in rows)
  assert db.graph_data_size_max == max(r.pickled_graph_tuple_size for r in rows)
  assert pytest.approx(
    db.graph_data_size_avg,
    sum(r.pickled_graph_tuple_size for r in rows) / len(rows),
  )

  if rows[0].data_flow_steps is None:
    assert db.data_flow_steps_min is None
    assert db.data_flow_steps_max is None
    assert db.data_flow_steps_avg is None
    assert db.data_flow_positive_node_count_min is None
    assert db.data_flow_positive_node_count_max is None
    assert db.data_flow_positive_node_count_avg is None
  else:
    assert db.data_flow_steps_min >= 0
    assert db.data_flow_steps_max >= 0
    assert db.data_flow_steps_avg >= 0
    assert db.data_flow_positive_node_count_min >= 0
    assert db.data_flow_positive_node_count_max >= 0
    assert db.data_flow_positive_node_count_avg >= 0


if __name__ == "__main__":
  test.Main()
