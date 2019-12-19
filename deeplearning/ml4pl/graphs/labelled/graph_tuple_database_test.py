# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //deeplearning/ml4pl/graphs/labelled:graph_tuple_database."""
import random

import pytest
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import random_graph_tuple_generator
from deeplearning.ml4pl.testing import random_networkx_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test


FLAGS = test.FLAGS

###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def empty_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def db_session(request) -> graph_tuple_database.Database.SessionType:
  """A test fixture which yields an empty graph proto database session."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    with db.Session() as session:
      yield session


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def two_graph_db_session(request) -> graph_tuple_database.Database.SessionType:
  """A test fixture which yields a database with two graph tuples."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
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


# Fixtures for enumerating populated databases.


@test.Fixture(scope="session", params=(500,))
def graph_count(request) -> int:
  return request.param


@test.Fixture(scope="session", params=(1, 3))
def node_x_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="session", params=(0, 3))
def node_y_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="session", params=(0, 3))
def graph_x_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="session", params=(0, 3))
def graph_y_dimensionality(request) -> int:
  return request.param


@test.Fixture(scope="session", params=(False, True))
def with_data_flow(request) -> bool:
  return request.param


@test.Fixture(scope="session", params=(0, 2))
def split_count(request) -> int:
  return request.param


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def populated_db_and_rows(
  request,
  graph_count: int,
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
  with_data_flow: bool,
  split_count: int,
) -> random_graph_tuple_database_generator.DatabaseAndRows:
  """Generate a populated database and a list of rows."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    yield random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count,
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      with_data_flow=with_data_flow,
      split_count=split_count,
    )


###############################################################################
# Tests.
###############################################################################


# CreateFromGraphTuple() tests.


@decorators.loop_for(seconds=5)
def test_CreateFromGraphTuple_attributes():
  """Test that attributes are copied over."""
  ir_id = random.randint(0, int(1e4))
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple()
  a = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
    graph_tuple, ir_id=ir_id
  )
  assert a.ir_id == ir_id
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


# Database stats tests.


@decorators.loop_for(min_iteration_count=3)
def test_database_stats_json_on_empty_db(
  empty_db: graph_tuple_database.Database,
):
  """Test computing stats on an empty database."""
  assert empty_db.stats_json


# Repeat test to use memoized property accessor.
@decorators.loop_for(min_iteration_count=3)
def test_database_stats_on_empty_database(
  populated_db_and_rows: random_graph_tuple_database_generator.DatabaseAndRows,
):
  """Test computing stats on an empty database."""
  db, _ = populated_db_and_rows
  assert db.stats_json


# Repeat test to use memoized property accessor.
@decorators.loop_for(min_iteration_count=3)
def test_database_stats_on_empty_db(empty_db: graph_tuple_database.Database):
  """Test accessing database stats on an empty database."""
  assert empty_db.graph_count == 0
  assert empty_db.ir_count == 0
  assert not empty_db.has_data_flow


# Repeat test to use memoized property accessor.
@decorators.loop_for(min_iteration_count=3)
def test_database_stats(
  populated_db_and_rows: random_graph_tuple_database_generator.DatabaseAndRows,
):
  """Test accessing database stats on a populated database."""
  db, rows = populated_db_and_rows

  # Graph and IR counts.
  assert db.graph_count == len(rows)
  assert db.ir_count == len(set(r.ir_id for r in rows))
  assert db.split_count <= len(set(r.split for r in rows))

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

  if rows[0].has_data_flow:
    assert db.has_data_flow
    assert db.data_flow_steps_min >= 0
    assert db.data_flow_steps_max >= 0
    assert db.data_flow_steps_avg >= 0
    assert db.data_flow_positive_node_count_min >= 0
    assert db.data_flow_positive_node_count_max >= 0
    assert db.data_flow_positive_node_count_avg >= 0
  else:
    assert not db.has_data_flow
    assert db.data_flow_steps_min is None
    assert db.data_flow_steps_max is None
    assert db.data_flow_steps_avg is None
    assert db.data_flow_positive_node_count_min is None
    assert db.data_flow_positive_node_count_max is None
    assert db.data_flow_positive_node_count_avg is None


###############################################################################
# Fuzzers.
###############################################################################


@decorators.loop_for(seconds=30)
def test_fuzz_GraphTuple_CreateFromGraphTuple(
  db_session: graph_tuple_database.Database.SessionType,
):
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  graph_tuple = random_graph_tuple_generator.CreateRandomGraphTuple()
  t = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
    graph_tuple=graph_tuple, ir_id=random.randint(0, int(4e6))
  )

  # Test the derived properties of the generated graph tuple.
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

  # Add it to the database to catch SQL integrity errors.
  db_session.add(t)
  db_session.commit()


@decorators.loop_for(seconds=30)
def test_fuzz_GraphTuple_CreateFromNetworkX(
  db_session: graph_tuple_database.Database.SessionType,
):
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  g = random_networkx_generator.CreateRandomGraph()
  t = graph_tuple_database.GraphTuple.CreateFromNetworkX(
    g=g, ir_id=random.randint(0, int(4e6))
  )

  # Test the derived properties of the generated graph tuple.
  assert t.edge_count == (
    t.control_edge_count + t.data_edge_count + t.call_edge_count
  )
  assert len(t.sha1) == 40
  assert t.node_count == g.number_of_nodes()
  assert t.edge_count == g.number_of_edges()
  assert t.tuple.node_count == g.number_of_nodes()
  assert t.tuple.edge_count == g.number_of_edges()
  assert len(t.tuple.adjacencies) == 3
  assert len(t.tuple.edge_positions) == 3

  # Add it to the database to catch SQL integrity errors.
  db_session.add(t)
  db_session.commit()


if __name__ == "__main__":
  test.Main()
