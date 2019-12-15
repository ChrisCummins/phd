"""This module defines a generator for databases of random graph tuples.

When executed as a script, this generates and populates a database of graphs:

    $ bazel run //deeplearning/ml4pl/testing:random_graph_tuple_database_generator -- \
        --graph_count=1000 \
        --graph_db='sqlite:////tmp/graphs.db'
"""
import copy
import random
from typing import List
from typing import NamedTuple

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.testing import generator_flags
from deeplearning.ml4pl.testing import random_graph_tuple_generator
from labm8.py import app

# This module is required to pull in FLAGS.
_unused_imports_ = generator_flags

FLAGS = app.FLAGS

app.DEFINE_integer("graph_count", 1000, "The number of graphs to generate.")
app.DEFINE_integer(
  "random_graph_pool_size",
  128,
  "The maximum number of random graphs to generate.",
)


def CreateRandomGraphTuple(
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  with_data_flow: bool = False,
  split_count: int = 0,
) -> graph_tuple_database.GraphTuple:
  """Create a random graph tuple."""
  mapped = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
    graph_tuple=random_graph_tuple_generator.CreateRandomGraphTuple(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
    ),
    ir_id=random.randint(0, int(4e6)),
    split=random.randint(0, split_count - 1) if split_count else None,
  )

  if with_data_flow:
    mapped.data_flow_steps = random.randint(1, 50)
    mapped.data_flow_root_node = random.randint(0, mapped.node_count - 1)
    mapped.data_flow_positive_node_count = random.randint(
      1, mapped.node_count - 1
    )

  return mapped


class DatabaseAndRows(NamedTuple):
  """"A graph tuple database and the rows that populate it."""

  db: graph_tuple_database.Database
  rows: List[graph_tuple_database.GraphTuple]


def PopulateDatabaseWithRandomGraphTuples(
  db: graph_tuple_database.Database,
  graph_count: int,
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  with_data_flow: bool = False,
  split_count: int = 0,
  random_graph_pool_size: int = 0,
) -> DatabaseAndRows:
  """Populate a database of random graph tuples."""
  random_graph_pool_size = random_graph_pool_size or min(
    FLAGS.random_graph_pool_size, 128
  )

  graph_pool = [
    CreateRandomGraphTuple(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      with_data_flow=with_data_flow,
      split_count=split_count,
    )
    for _ in range(random_graph_pool_size)
  ]

  # Generate a full list of graph rows by randomly selecting from the graph
  # pool.
  rows = [copy.deepcopy(random.choice(graph_pool)) for _ in range(graph_count)]

  with db.Session(commit=True) as session:
    session.add_all([copy.deepcopy(t) for t in rows])

  db.RefreshStats()

  return DatabaseAndRows(db, rows)


def Main():
  """Main entry point"""
  graph_db = FLAGS.graph_db()
  PopulateDatabaseWithRandomGraphTuples(
    graph_db,
    FLAGS.graph_count,
    node_x_dimensionality=FLAGS.node_x_dimensionality,
    node_y_dimensionality=FLAGS.node_y_dimensionality,
    graph_x_dimensionality=FLAGS.graph_x_dimensionality,
    graph_y_dimensionality=FLAGS.graph_y_dimensionality,
    with_data_flow=FLAGS.with_data_flow,
    split_count=FLAGS.split_count,
  )


if __name__ == "__main__":
  app.Run(Main)
