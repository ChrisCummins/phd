"""Unit tests for //deeplearning/ml4pl/testing:random_unlabelled_graph_database_generator."""
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.testing import (
  random_unlabelled_graph_database_generator,
)
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@decorators.loop_for(seconds=2)
@test.Parametrize("node_x_dimensionality", (1, 2))
@test.Parametrize("node_y_dimensionality", (0, 1, 2))
@test.Parametrize("graph_x_dimensionality", (0, 1, 2))
@test.Parametrize("graph_y_dimensionality", (0, 1, 2))
def test_CreateRandomProgramGraph(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
):
  """Black-box test of generator properties."""
  program_graph = random_unlabelled_graph_database_generator.CreateRandomProgramGraph(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )
  assert program_graph.node_x_dimensionality == node_x_dimensionality
  assert program_graph.node_y_dimensionality == node_y_dimensionality
  assert program_graph.graph_x_dimensionality == graph_x_dimensionality
  assert program_graph.graph_y_dimensionality == graph_y_dimensionality


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("db"),
)
def db(request) -> unlabelled_graph_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    unlabelled_graph_database.Database, request.param
  )


@test.Fixture(scope="function", params=(1, 1000, 5000))
def proto_count(request) -> int:
  """Test fixture to enumerate proto counts."""
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


@test.Fixture(scope="function", params=(0, 2))
def split_count(request) -> int:
  """Test fixture to enumerate split counts."""
  return request.param


def test_PopulateDatabaseWithRandomProgramGraphs(
  db: unlabelled_graph_database.Database,
  proto_count: int,
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
  split_count: int,
):
  """Test populating databases."""
  random_unlabelled_graph_database_generator.PopulateDatabaseWithRandomProgramGraphs(
    db=db,
    proto_count=proto_count,
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    split_count=split_count,
  )
  with db.Session() as session:
    assert (
      session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
      ).scalar()
      == proto_count
    )

    assert (
      session.query(
        sql.func.min(
          unlabelled_graph_database.ProgramGraph.node_x_dimensionality
        )
      ).scalar()
      == node_x_dimensionality
    )

    assert (
      session.query(
        sql.func.min(
          unlabelled_graph_database.ProgramGraph.node_y_dimensionality
        )
      ).scalar()
      == node_y_dimensionality
    )

    assert (
      session.query(
        sql.func.min(
          unlabelled_graph_database.ProgramGraph.graph_y_dimensionality
        )
      ).scalar()
      == graph_y_dimensionality
    )

    assert (
      session.query(
        sql.func.min(
          unlabelled_graph_database.ProgramGraph.graph_y_dimensionality
        )
      ).scalar()
      == graph_y_dimensionality
    )


def test_benchmark_CreateRandomProgramGraph(benchmark):
  """Benchmark graph tuple generation."""
  benchmark(random_unlabelled_graph_database_generator.CreateRandomProgramGraph)


if __name__ == "__main__":
  test.Main()
