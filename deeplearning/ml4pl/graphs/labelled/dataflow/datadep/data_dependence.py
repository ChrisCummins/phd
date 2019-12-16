"""Module for labelling program graphs with data depedencies."""
import collections

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app

FLAGS = app.FLAGS


# The node_x arrays for node data dependencies:
NOT_DEPENDENCY = [1, 0]
DEPENDENCY = [0, 1]


class DataDependencyAnnotator(data_flow_graphs.NetworkXDataFlowGraphAnnotator):
  """Annotate graphs with data dependencies.

  Statement node A depends on statement B iff B produces data nodes that are
  operands to A.
  """

  def IsValidRootNode(self, node: int, data) -> bool:
    """Data dependency is a statement-based analysis."""
    return data["type"] == programl_pb2.Node.STATEMENT and data["function"]

  def Annotate(self, g: nx.MultiDiGraph, root_node: int) -> None:
    """Annotate all of the nodes that must be executed prior to the root node.
    """
    # Initialize all nodes as not dependent and not the root node, except the root
    # node.
    for node, data in g.nodes(data=True):
      data["x"].append(data_flow_graphs.ROOT_NODE_NO)
      data["y"] = NOT_DEPENDENCY
    g.nodes[root_node]["x"][-1] = data_flow_graphs.ROOT_NODE_YES

    # Breadth-first traversal to mark node dependencies.
    data_flow_steps = 0
    dependency_node_count = 0
    visited = set()
    q = collections.deque([(root_node, 1)])
    while q:
      next, data_flow_steps = q.popleft()
      dependency_node_count += 1
      visited.add(next)

      # Mark the node as dependent.
      g.nodes[next]["y"] = DEPENDENCY

      # Visit all data predecessors.
      for pred, _, flow in g.in_edges(next, data="flow"):
        if flow == programl_pb2.Edge.DATA and pred not in visited:
          q.append((pred, data_flow_steps + 1))

    g.graph["data_flow_root_node"] = root_node
    g.graph["data_flow_steps"] = data_flow_steps
    g.graph["data_flow_positive_node_count"] = dependency_node_count
