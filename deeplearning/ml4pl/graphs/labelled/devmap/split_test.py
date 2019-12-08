"""Unit tests for //deeplearning/ml4pl/graphs/labelled/devmap:split."""
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import split
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def empty_graph_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with random graph tuples."""
  yield testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def populated_graph_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with random graph tuples."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db, graph_count=100, graph_y_dimensionality=2
    )
    yield db


@test.Parametrize("k", (3, 5))
@decorators.loop_for(seconds=5, min_iteration_count=3)
def test_fuzz(
  populated_graph_db: graph_tuple_database.Database,
  k: int,
  empty_graph_db: graph_tuple_database.Database,
):
  """Opaque fuzzing of the public methods."""
  splitter = split.StratifiedGraphLabelKFold(k)
  splitter.ApplySplit(populated_graph_db)
  split.CopySplits(populated_graph_db, empty_graph_db)


if __name__ == "__main__":
  test.Main()
