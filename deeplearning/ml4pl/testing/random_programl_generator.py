"""This module defines a generator for random program graph protos."""
import functools
import pickle
import random
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
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
  return [random.randint(lower_value, upper_value) for _ in range(n)]


def CreateRandomProto(
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  node_count: int = None,
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
  node_count = node_count or random.randint(5, 50)

  if node_count < 2:
    raise ValueError("node_count < 2")

  proto = programl_pb2.ProgramGraph()

  def _RandomDst(src: int) -> int:
    """Select a random destination node for the given source."""
    n = random.randint(0, node_count - 1)
    if n == src:
      return _RandomDst(src)
    else:
      return n

  functions = []

  # Create the nodes.
  for i in range(node_count):
    node = proto.node.add()
    if i:
      node.type = np.random.choice(
        [
          programl_pb2.Node.STATEMENT,
          programl_pb2.Node.IDENTIFIER,
          programl_pb2.Node.IMMEDIATE,
        ],
        p=[0.45, 0.3, 0.25],
      )
      if node.type == programl_pb2.Node.STATEMENT:
        node.text = "statement"
        node.preprocessed_text = "!UNK"
        if functions and random.random() < 0.85:
          node.function = random.choice(functions)
        else:
          node.function = len(functions)
        functions.append(len(functions))
      elif node.type == programl_pb2.Node.IDENTIFIER:
        node.text = "%0"
        node.preprocessed_text = "!IDENTIFIER"
      else:
        node.text = "0"
        node.preprocessed_text = "!IDENTIFIER"
    else:
      # The first node is always the root.
      node.type = programl_pb2.Node.STATEMENT
      node.text = "root"
      node.preprocessed_text = "!UNK"

    # Add the node features and labels.
    node.x[:] = _CreateRandomList(node_x_dimensionality)
    if node_y_dimensionality:
      node.y[:] = _CreateRandomList(node_y_dimensionality)

  # Create the functions.
  for i in functions:
    function = proto.function.add()
    function.name = f"fn_{i}"

  # Keep track of the edges that we have created to avoid generating parallel
  # edges of the same flow.
  edges: Set[Tuple[int, int, programl_pb2.Edge.Flow]] = set()

  # Create the edges.
  for src, node in enumerate(proto.node):
    outgoing_edge_count = random.randint(1, 3)
    for _ in range(outgoing_edge_count):
      dst = _RandomDst(src)

      # Determine the flow based on the source node type.
      if src:
        if node.type == programl_pb2.Node.STATEMENT:
          flow = np.random.choice(
            [programl_pb2.Edge.CONTROL, programl_pb2.Edge.CALL], p=[0.9, 0.1]
          )
        else:
          flow = programl_pb2.Edge.DATA
      else:
        flow = programl_pb2.Edge.CALL

      if (src, dst, flow) not in edges:
        edges.add((src, dst, flow))

        edge = proto.edge.add()
        edge.flow = flow
        edge.source_node = src
        edge.destination_node = dst
        edge.position = random.randint(0, 4)

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
