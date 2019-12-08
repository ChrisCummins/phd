"""Unit tests for //deeplearning/ml4pl/testing:random_graph_tuple_database_generator."""
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@decorators.loop_for(seconds=2)
@test.Parametrize("node_x_dimensionality", (1, 2))
@test.Parametrize("node_y_dimensionality", (0, 1, 2))
@test.Parametrize("graph_x_dimensionality", (0, 1, 2))
@test.Parametrize("graph_y_dimensionality", (0, 1, 2))
@test.Parametrize("with_data_flow", (False, True))
def test_CreateRandomGraphTuple(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
  with_data_flow: bool,
):
  """Black-box test of generator properties."""
  graph_tuple = random_graph_tuple_database_generator.CreateRandomGraphTuple(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    with_data_flow=with_data_flow,
  )
  assert graph_tuple.node_x_dimensionality == node_x_dimensionality
  assert graph_tuple.node_y_dimensionality == node_y_dimensionality
  assert graph_tuple.graph_x_dimensionality == graph_x_dimensionality
  assert graph_tuple.graph_y_dimensionality == graph_y_dimensionality
  if with_data_flow:
    assert graph_tuple.data_flow_steps >= 1
    assert graph_tuple.data_flow_root_node >= 0
    assert graph_tuple.data_flow_positive_node_count >= 1
  else:
    assert graph_tuple.data_flow_steps is None
    assert graph_tuple.data_flow_root_node is None
    assert graph_tuple.data_flow_positive_node_count is None


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def db(request) -> graph_tuple_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


@test.Fixture(scope="function", params=(1, 1000, 5000))
def graph_count(request) -> int:
  """Test fixture to enumerate graph counts."""
  return request.param


@test.Fixture(scope="function", params=(1, 3))
def node_x_dimensionality(request) -> int:
  """Test fixture to enumerate node feature dimensionalities."""
  return request.param


@test.Fixture(scope="function", params=(0, 3))
def node_y_dimensionality(request) -> int:
  """Test fixture to enumerate node label dimensionalities."""
  return request.param


@test.Fixture(scope="function", params=(0, 3))
def graph_x_dimensionality(request) -> int:
  """Test fixture to enumerate graph feature dimensionalities."""
  return request.param


@test.Fixture(scope="function", params=(0, 3))
def graph_y_dimensionality(request) -> int:
  """Test fixture to enumerate graph label dimensionalities."""
  return request.param


@test.Fixture(scope="function", params=(False, True))
def with_data_flow(request) -> int:
  """Test fixture to enumerate data flows."""
  return request.param


@test.Fixture(scope="function", params=(0, 2))
def split_count(request) -> int:
  """Test fixture to enumerate split counts."""
  return request.param


def test_PopulateDatahaseithRandomGraphTuples(
  db: graph_tuple_database.Database,
  graph_count: int,
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
  with_data_flow: bool,
  split_count: int,
):
  """Test populating databases."""
  random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
    db=db,
    graph_count=graph_count,
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    with_data_flow=with_data_flow,
    split_count=split_count,
  )
  with db.Session() as session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      == graph_count
    )

    assert (
      session.query(
        sql.func.min(graph_tuple_database.GraphTuple.node_x_dimensionality)
      ).scalar()
      == node_x_dimensionality
    )

    assert (
      session.query(
        sql.func.min(graph_tuple_database.GraphTuple.node_y_dimensionality)
      ).scalar()
      == node_y_dimensionality
    )

    assert (
      session.query(
        sql.func.min(graph_tuple_database.GraphTuple.graph_y_dimensionality)
      ).scalar()
      == graph_y_dimensionality
    )

    assert (
      session.query(
        sql.func.min(graph_tuple_database.GraphTuple.graph_y_dimensionality)
      ).scalar()
      == graph_y_dimensionality
    )


def test_benchmark_CreateRandomGraphTuple(benchmark):
  """Benchmark graph tuple generation."""
  benchmark(random_graph_tuple_database_generator.CreateRandomGraphTuple)


if __name__ == "__main__":
  test.Main()
