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

from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.testing import random_graph_tuple_generator
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_integer("graph_count", 1000, "The number of graphs to generate.")
app.DEFINE_integer(
  "node_x_dimensionality", 1, "The dimensionality of node x vectors."
)
app.DEFINE_integer(
  "node_y_dimensionality", 1, "The dimensionality of node y vectors."
)
app.DEFINE_integer(
  "graph_x_dimensionality", 1, "The dimensionality of graph x vectors."
)
app.DEFINE_integer(
  "graph_y_dimensionality", 1, "The dimensionality of graph y vectors."
)


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
) -> graph_tuple.GraphTuple:
  """Populate a database of random graph tuples."""
  graph_pool = [
    graph_tuple_database.GraphTuple.CreateFromGraphTuple(
      graph_tuple=random_graph_tuple_generator.CreateRandomGraphTuple(
        node_x_dimensionality=node_x_dimensionality,
        node_y_dimensionality=node_y_dimensionality,
        graph_x_dimensionality=graph_x_dimensionality,
        graph_y_dimensionality=graph_y_dimensionality,
      ),
      ir_id=random.randint(0, int(4e6)),
    )
    for _ in range(min(graph_count, 128))
  ]

  # Generate a full list of graph rows by randomly selecting from the graph
  # pool.
  rows = [copy.deepcopy(random.choice(graph_pool)) for _ in range(graph_count)]

  with db.Session(commit=True) as session:
    session.add_all([copy.deepcopy(t) for t in rows])

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
  )


if __name__ == "__main__":
  app.Run(Main)
