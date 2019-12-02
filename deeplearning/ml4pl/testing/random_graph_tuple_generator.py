"""This module defines a generator for random graph tuples."""
import functools
from typing import Iterable

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test

FLAGS = test.FLAGS


def CreateRandomGraphTuple(
  disjoint_graph_count: int = 1,
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
) -> graph_tuple.GraphTuple:
  """Generate a random graph tuple.

  This generates a random graph which has sensible values for fields, but does
  not have meaningful semantics, e.g. there may be data flow edges between
  identifiers, etc. For speed, this generator guarantees only that:

    1. There is a 'root' node with outgoing call edges.
    2. Nodes are either statements, identifiers, or immediates.
    3. Nodes have text, preprocessed_text, and a single node_x value.
    4. Edges are either control, data, or call.
    5. Edges have positions.
    6. The graph is strongly connected.
  """
  protos = [
    random_programl_generator.CreateRandomProto(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
    )
    for _ in range(disjoint_graph_count)
  ]

  graphs = [programl.ProgramGraphToNetworkX(proto) for proto in protos]
  graph_tuples = [
    graph_tuple.GraphTuple.CreateFromNetworkX(graph) for graph in graphs
  ]
  if len(graph_tuples) > 1:
    return graph_tuple.GraphTuple.FromGraphTuples(graph_tuples)
  else:
    return graph_tuples[0]


@functools.lru_cache(maxsize=2)
def EnumerateGraphTupleTestSet() -> Iterable[graph_tuple.GraphTuple]:
  """Enumerate a test set of "real" graph tuples."""
  for proto in random_programl_generator.EnumerateProtoTestSet():
    g = programl.ProgramGraphToNetworkX(proto)
    yield graph_tuple.GraphTuple.CreateFromNetworkX(g)
