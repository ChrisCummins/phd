"""This module defines a generator for databases of unlabelled graphs.

When executed as a script, this generates and populates a database of graphs:

    $ bazel run //deeplearning/ml4pl/testing:random_unlabelled_graph_database_generator -- \
        --proto_count=1000 \
        --proto_db='sqlite:////tmp/protos.db'
"""
import copy
import itertools
import random
from typing import List
from typing import NamedTuple
from typing import Optional

from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.testing import generator_flags
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import app

# This module is required to pull in FLAGS.
_unused_imports_ = generator_flags

FLAGS = app.FLAGS

app.DEFINE_integer("proto_count", 1000, "The number of graphs to generate.")
app.DEFINE_integer(
  "random_proto_pool_size",
  128,
  "The maximum number of random protos to generate.",
)


def CreateRandomProgramGraph(
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  split_count: int = 0,
) -> unlabelled_graph_database.ProgramGraph:
  """Create a random unlabelled graph."""
  return unlabelled_graph_database.ProgramGraph.Create(
    proto=random_programl_generator.CreateRandomProto(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      with_data_flow=False,
    ),
    # Note that ir_id is a primary key, so you may need to set this value
    # explicitly to avoid integrity errors.
    ir_id=random.randint(0, int(4e6)),
    split=random.randint(0, split_count) if split_count else None,
  )


class DatabaseAndRows(NamedTuple):
  """"A graph tuple database and the rows that populate it."""

  db: unlabelled_graph_database.Database
  rows: List[unlabelled_graph_database.ProgramGraph]


def PopulateDatabaseWithRandomProgramGraphs(
  db: unlabelled_graph_database.Database,
  proto_count: int,
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  split_count: int = 0,
  random_proto_pool_size: int = 0,
) -> DatabaseAndRows:
  """Populate a database of random graph tuples."""
  random_proto_pool_size = random_proto_pool_size or min(
    FLAGS.random_proto_pool_size, 128
  )

  graph_pool = [
    CreateRandomProgramGraph(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      split_count=split_count,
    )
    for _ in range(random_proto_pool_size)
  ]

  # Generate a full list of rows by randomly selecting from the graph pool.
  rows = [copy.deepcopy(random.choice(graph_pool)) for _ in range(proto_count)]

  # Assign unique keys.
  for i, row in enumerate(rows):
    row.ir_id = i + 1

  with db.Session(commit=True) as session:
    session.add_all([copy.deepcopy(t) for t in rows])

  return DatabaseAndRows(db, rows)


def PopulateDatabaseWithTestSet(
  db: unlabelled_graph_database.Database, graph_count: Optional[int] = None
):
  """Populate a database with "real" programs."""
  inputs = itertools.islice(
    itertools.cycle(random_programl_generator.EnumerateTestSet(n=graph_count)),
    graph_count,
  )

  with db.Session(commit=True) as session:
    session.add_all(
      [
        unlabelled_graph_database.ProgramGraph.Create(proto, ir_id=i + 1)
        for i, proto in enumerate(inputs)
      ]
    )
  return db


def Main():
  """Main entry point"""
  PopulateDatabaseWithRandomProgramGraphs(
    FLAGS.proto_db(),
    FLAGS.proto_count,
    node_x_dimensionality=FLAGS.node_x_dimensionality,
    node_y_dimensionality=FLAGS.node_y_dimensionality,
    graph_x_dimensionality=FLAGS.graph_x_dimensionality,
    graph_y_dimensionality=FLAGS.graph_y_dimensionality,
    split_count=FLAGS.split_count,
  )


if __name__ == "__main__":
  app.Run(Main)
