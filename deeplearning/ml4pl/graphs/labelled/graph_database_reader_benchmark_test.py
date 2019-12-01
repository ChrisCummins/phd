"""Benchmarks for //deeplearning/ml4pl/graphs/labelled:graph_database_reader."""
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

MODULE_UNDER_TEST = None


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


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
def test_benchmark_BufferedGraphReader_in_order(
  benchmark, db_10000: graph_tuple_database.Database, buffer_size_mb: int,
):
  """Benchmark in-order database reads."""
  benchmark(
    list,
    reader.BufferedGraphReader(
      db_10000,
      buffer_size_mb=buffer_size_mb,
      order=reader.BufferedGraphReaderOrder.IN_ORDER,
    ),
  )


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
def test_benchmark_BufferedGraphReader_global_random(
  benchmark, db_10000: graph_tuple_database.Database, buffer_size_mb: int,
):
  """Benchmark global random database reads."""
  benchmark(
    list,
    reader.BufferedGraphReader(
      db_10000,
      buffer_size_mb=buffer_size_mb,
      order=reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
    ),
  )


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
def test_benchmark_BufferedGraphReader_batch_random(
  benchmark, db_10000: graph_tuple_database.Database, buffer_size_mb: int,
):
  """Benchmark batch random database reads."""
  benchmark(
    list,
    reader.BufferedGraphReader(
      db_10000,
      buffer_size_mb=buffer_size_mb,
      order=reader.BufferedGraphReaderOrder.BATCH_RANDOM,
    ),
  )


@test.Parametrize("buffer_size_mb", READER_BUFFER_SIZES)
def test_benchmark_BufferedGraphReader_data_flow_steps(
  benchmark, db_10000: graph_tuple_database.Database, buffer_size_mb: int,
):
  """Benchmark ordered database reads."""
  benchmark(
    list,
    reader.BufferedGraphReader(
      db_10000,
      buffer_size_mb=buffer_size_mb,
      order=reader.BufferedGraphReaderOrder.DATA_FLOW_STEPS,
    ),
  )


if __name__ == "__main__":
  test.Main()
