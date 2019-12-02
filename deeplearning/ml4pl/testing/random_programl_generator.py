"""This module defines a generator for random program graph protos."""
import functools
import pickle
import random
from typing import Iterable
from typing import List

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from labm8.py import bazelutil
from labm8.py import test

FLAGS = test.FLAGS

# A test set of unlabelled program graphs using the legacy networkx schema.
# These must be migrated to the new program graph representation before use.
NETWORKX_GRAPHS_ARCHIVE = bazelutil.DataArchive(
  "phd/deeplearning/ml4pl/testing/data/100_unlabelled_networkx_graphs.tar.bz2"
)


def _CreateRandomList(
  n: int, lower_value: int = 0, upper_value: int = 10
) -> List[int]:
  """Generate 'n' random ints in the range [lower_value, upper_value]."""
  return [random.randint(lower_value, upper_value + 1) for _ in range(n)]


def CreateRandomProto(
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
) -> programl_pb2.ProgramGraph:
  """Generate a random program graph.

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
  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)

  if node_x_dimensionality < 1:
    raise ValueError("node_x_dimensionality < 1")

  if node_x_dimensionality > 1:
    for node in proto.node:
      node.x.extend(_CreateRandomList(node_x_dimensionality - 1))

  if node_y_dimensionality:
    for node in proto.node:
      node.y[:] = _CreateRandomList(node_y_dimensionality)

  if graph_x_dimensionality:
    proto.x[:] = _CreateRandomList(graph_x_dimensionality)

  if graph_y_dimensionality:
    proto.y[:] = _CreateRandomList(graph_y_dimensionality)

  return proto


@functools.lru_cache(maxsize=2)
def EnumerateProtoTestSet() -> Iterable[programl_pb2.ProgramGraph]:
  """Enumerate a test set of "real" program graphs."""
  with NETWORKX_GRAPHS_ARCHIVE as pickled_dir:
    for path in pickled_dir.iterdir():
      with open(path, "rb") as f:
        old_nx_graph = pickle.load(f)
        yield networkx_to_protos.NetworkXGraphToProgramGraphProto(old_nx_graph)
