"""Benchmarks for //deeplearning/ml4pl/ggnn:graph_database_reader."""
import pickle
import random

import numpy as np

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as reader
from labm8.py import labtypes
from labm8.py import test

FLAGS = test.FLAGS

MODULE_UNDER_TEST = None


def PopulateGraphDatabase(
  graph_db: graph_database.Database, graph_count: int
) -> graph_database.Database:
  """Populate a graph database with the requested number of random graphs."""

  def MakeRandomGraph():
    """Make a random graph."""
    return graph_database.GraphMeta(
      group=random.choice(["train", "val", "test"]),
      bytecode_id=random.randint(0, 100000000),
      source_name="foo",
      relpath="bar",
      language=random.choice(["c", "opencl", "swift", "haskell"]),
      node_count=random.randint(0, 100000),
      edge_count=random.randint(0, 200000),
      edge_position_max=random.randint(0, 100),
      data_flow_max_steps_required=random.randint(0, 1000),
      loop_connectedness=0,
      undirected_diameter=0,
      graph=graph_database.Graph(
        # ~200KB of pickled data
        pickled_data=pickle.dumps(np.random.rand(200000 // 4))
      ),
    )

  with graph_db.Session(commit=True) as s:
    for subrange in labtypes.Chunkify(list(range(graph_count)), 512):
      s.add_all([MakeRandomGraph() for _ in range(len(subrange))])

  return graph_db


@test.Parametrize("buffer_size", [128, 512])
@test.Parametrize("graph_count", [1000, 10000])
def test_benchmark_BufferedGraphReader_in_order(
  benchmark,
  empty_graph_db: graph_database.Database,
  buffer_size: int,
  graph_count: int,
):
  """Benchmark in-order database reads."""
  graph_db = PopulateGraphDatabase(empty_graph_db, graph_count)
  benchmark(
    list,
    reader.BufferedGraphReader(
      graph_db,
      buffer_size=buffer_size,
      order=reader.BufferedGraphReaderOrder.IN_ORDER,
    ),
  )


@test.Parametrize("buffer_size", [128, 512])
@test.Parametrize("graph_count", [1000, 10000])
def test_benchmark_BufferedGraphReader_global_random(
  benchmark,
  empty_graph_db: graph_database.Database,
  buffer_size: int,
  graph_count: int,
):
  """Benchmark global random database reads."""
  graph_db = PopulateGraphDatabase(empty_graph_db, graph_count)
  benchmark(
    list,
    reader.BufferedGraphReader(
      graph_db,
      buffer_size=buffer_size,
      order=reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
    ),
  )


@test.Parametrize("buffer_size", [128, 512])
@test.Parametrize("graph_count", [1000, 10000])
def test_benchmark_BufferedGraphReader_batch_random(
  benchmark,
  empty_graph_db: graph_database.Database,
  buffer_size: int,
  graph_count: int,
):
  """Benchmark batch random database reads."""
  graph_db = PopulateGraphDatabase(empty_graph_db, graph_count)
  benchmark(
    list,
    reader.BufferedGraphReader(
      graph_db,
      buffer_size=buffer_size,
      order=reader.BufferedGraphReaderOrder.BATCH_RANDOM,
    ),
  )


@test.Parametrize("buffer_size", [128, 512])
@test.Parametrize("graph_count", [1000, 10000])
def test_benchmark_BufferedGraphReader_data_flow_max_steps_required(
  benchmark,
  empty_graph_db: graph_database.Database,
  buffer_size: int,
  graph_count: int,
):
  """Benchmark ordered database reads."""
  graph_db = PopulateGraphDatabase(empty_graph_db, graph_count)
  benchmark(
    list,
    reader.BufferedGraphReader(
      graph_db,
      buffer_size=buffer_size,
      order=reader.BufferedGraphReaderOrder.DATA_FLOW_MAX_STEPS_REQUIRED,
    ),
  )


if __name__ == "__main__":
  test.Main()
