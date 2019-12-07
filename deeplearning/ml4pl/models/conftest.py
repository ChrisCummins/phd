"""Test fixtures for models."""
from typing import Iterable
from typing import Tuple

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logging
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def log_db(request) -> log_database.Database:
  """A test fixture which yields an empty log database."""
  yield from testing_databases.YieldDatabase(
    log_database.Database, request.param
  )


@test.Fixture(scope="function")
def logger(log_db: log_database.Database) -> logging.Logger:
  """A test fixture which yields a log database."""
  # TODO: Parameterize logger.
  with logging.Logger(
    log_db,
    max_buffer_size=None,
    max_buffer_length=128,
    max_seconds_since_flush=None,
  ) as logger:
    yield logger


@test.Fixture(scope="session", params=((0, 2), (0, 3), (2, 0), (3, 0)))
def dimensionalities(request) -> Tuple[int, int]:
  """A test fixture which enumerates node and graph label dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(0, 3))
def edge_position_max(request) -> int:
  """A test fixture which enumerates edge positions."""
  return request.param


@test.Fixture(scope="session", params=(None, "foo"))
def restored_from(request) -> str:
  """A test fixture for 'restored_from' values."""
  return request.param


@test.Fixture(scope="session", params=list(epoch.Type))
def epoch_type(request) -> epoch.Type:
  """A test fixture which enumerates epoch types."""
  return request.param


# Currently, only 2-dimension node features are supported.
@test.Fixture(scope="session", params=(2,))
def node_x_dimensionality(request) -> int:
  """A test fixture which enumerates node feature dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(True,))
def with_data_flow(request) -> int:
  """A test fixture which enumerates 'with dataflow' values."""
  return request.param


@test.Fixture(scope="session", params=(10, 100))
def graph_count(request) -> int:
  """A test fixture which enumerates graph counts."""
  return request.param


@test.Fixture(scope="session")
def graphs(
  graph_count: int,
  dimensionalities: Tuple[int, int],
  node_x_dimensionality: int,
  with_data_flow: bool,
) -> Iterable[graph_tuple_database.GraphTuple]:
  """A test fixture which enumerates graph tuples."""
  node_y_dimensionality, graph_y_dimensionality = dimensionalities

  def MakeIterator(list):
    yield from list

  return MakeIterator(
    [
      random_graph_tuple_database_generator.CreateRandomGraphTuple(
        node_x_dimensionality=node_x_dimensionality,
        node_y_dimensionality=node_y_dimensionality,
        graph_y_dimensionality=graph_y_dimensionality,
        with_data_flow=with_data_flow,
      )
      for _ in range(graph_count)
    ]
  )
