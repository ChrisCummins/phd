"""Unit tests //deeplearning/ml4pl/graphs/labelled:graph_tuple_database_reader"""
import copy
import random
from typing import List

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.labelled import graph_database_reader as reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test

FLAGS = test.FLAGS


def CreateRandomGraphTuple(ir_id: int) -> graph_tuple_database.GraphTuple:
  """Generate a random graph tuple."""
  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)
  g = programl.ProgramGraphToNetworkX(proto)
  gt = graph_tuple.GraphTuple.CreateFromNetworkX(g)
  return graph_tuple_database.GraphTuple.CreateFromGraphTuple(gt, ir_id=ir_id)


@test.Fixture(scope="session", params=testing_databases.TEST_DB_URLS)
def empty_graph_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields an empty database."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


@test.Fixture(scope="session")
def db_10000(
  empty_graph_db: graph_tuple_database.Database,
) -> graph_tuple_database.Database:
  """Fixture which returns a database with 5000 + 2 graph tuples, where 2 of the
  graph tuples are empty.

  For the current implementation of CreateRandomGraphTuple(), a database of
  5000 graphs is ~14MB of data.
  """
  # Generate some random graph tuples.
  graph_pool = [CreateRandomGraphTuple(0) for _ in range(128)]

  # Generate a full list of graphs by randomly selecting from the graph pool.
  random_graph_tuples: List[graph_tuple_database.GraphTuple] = [
    copy.deepcopy(random.choice(graph_pool)) for _ in range(10000)
  ]
  # Index the random graphs by ir_id.
  for i, t in enumerate(random_graph_tuples):
    t.ir_id = i
    t.data_flow_steps = i

  with empty_graph_db.Session(commit=True) as s:
    s.add_all(random_graph_tuples)
    # Create the empty graph tuples. These should be ignored by the graph
    # reader.
    s.add_all(
      [
        graph_tuple_database.GraphTuple.CreateEmpty(0),
        graph_tuple_database.GraphTuple.CreateEmpty(0),
      ]
    )

  return empty_graph_db


# Buffer sizes selected based on the size of the db_1000 fixture database.
READER_BUFFER_SIZES = [1, 4, 1024]
# The full list of graph reader orders.
ALL_READER_ORDERS = [
  reader.BufferedGraphReaderOrder.IN_ORDER,
  reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
  reader.BufferedGraphReaderOrder.BATCH_RANDOM,
  reader.BufferedGraphReaderOrder.DATA_FLOW_STEPS,
]
# The list of random-order readers.
RANDOM_READER_ORDERS = [
  reader.BufferedGraphReaderOrder.BATCH_RANDOM,
  reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
]


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
def test_BufferedGraphReader_values_in_order(
  db_10000: graph_tuple_database.Database, buffer_size_mb: int
):
  """Test that the expected number of graphs are returned"""
  graphs = list(
    reader.BufferedGraphReader(db_10000, buffer_size_mb=buffer_size_mb)
  )
  assert len(graphs) == 10000
  # When read in order, the ir_ids should be equal to their position.
  assert all([g.ir_id == i for i, g in enumerate(graphs)])


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
@test.Parametrize("order", ALL_READER_ORDERS)
def test_BufferedGraphReader_filters(
  db_10000: graph_tuple_database.Database,
  buffer_size_mb: int,
  order: reader.BufferedGraphReaderOrder,
):
  """Test when using filters to limit results."""
  filters = [
    lambda: graph_tuple_database.GraphTuple.ir_id < 3000,
    lambda: graph_tuple_database.GraphTuple.data_flow_steps < 2000,
  ]
  graphs = list(
    reader.BufferedGraphReader(
      db_10000, filters=filters, buffer_size_mb=buffer_size_mb, order=order
    )
  )
  assert len(graphs) == 2000


@test.Flaky(reason="Random order may produce rows in order")
@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
@test.Parametrize("order", RANDOM_READER_ORDERS)
def test_BufferedGraphReader_random_orders(
  db_10000: graph_tuple_database.Database,
  buffer_size_mb: int,
  order: reader.BufferedGraphReaderOrder,
):
  """Test that random order return rows in randomized order."""
  graphs = list(
    reader.BufferedGraphReader(
      db_10000, buffer_size_mb=buffer_size_mb, order=order
    )
  )
  ir_ids = [g.ir_id for g in graphs]
  assert ir_ids != sorted(ir_ids)


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
@test.Parametrize("order", ALL_READER_ORDERS)
@test.Parametrize("limit", [1, 25, 9999999])
def test_BufferedGraphReader_limit(
  db_10000: graph_tuple_database.Database,
  buffer_size_mb: int,
  order: reader.BufferedGraphReaderOrder,
  limit: int,
):
  """Test using `limit` arg to limit number of returned rows."""
  graphs = list(
    reader.BufferedGraphReader(
      db_10000, limit=limit, buffer_size_mb=buffer_size_mb, order=order
    )
  )
  assert len(graphs) == min(limit, 10000)


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
@test.Parametrize("order", ALL_READER_ORDERS)
def test_BufferedGraphReader_next(
  db_10000: graph_tuple_database.Database,
  buffer_size_mb: int,
  order: reader.BufferedGraphReaderOrder,
):
  """Test using next() to read from BufferedGraphReader()."""
  db_reader = reader.BufferedGraphReader(
    db_10000, buffer_size_mb=buffer_size_mb, order=order
  )
  for _ in range(10000):
    next(db_reader)
  with test.Raises(StopIteration):
    next(db_reader)


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
def test_BufferedGraphReader_data_flow_steps_order(
  db_10000: graph_tuple_database.Database, buffer_size_mb: int
):
  """Test that data flow max steps increases monotonically."""
  db_reader = reader.BufferedGraphReader(
    db_10000,
    buffer_size_mb=buffer_size_mb,
    order=reader.BufferedGraphReaderOrder.DATA_FLOW_STEPS,
  )
  current_steps = -1
  i = 0
  for i, graph in enumerate(db_reader):
    # Sanity check that test fixture set expected values for data flow steps.
    assert graph.data_flow_steps == graph.ir_id
    # Assert that data flow steps is monotonically increasing.
    assert graph.data_flow_steps >= current_steps
    current_steps = graph.data_flow_steps
  # Sanity check that the correct number of graphs were returned.
  assert i + 1 == 10000


if __name__ == "__main__":
  test.Main()
