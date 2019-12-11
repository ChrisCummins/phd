"""Module for labelling program graphs with dominator trees information."""
from typing import Dict
from typing import List
from typing import Set

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app

FLAGS = app.FLAGS


# The real_y arrays for node reachability:
NOT_DOMINATED = [1, 0]
DOMINATED = [0, 1]


class DominatorTreeAnnotator(data_flow_graphs.NetworkXDataFlowGraphAnnotator):
  """Annotate graphs with dominator analysis.

  Statement node A dominates statement node B iff all control paths to B pass
  through A.
  """

  def RootNodeType(self) -> programl_pb2.Node.Type:
    """Dominator trees are a statement-based analysis."""
    return programl_pb2.Node.STATEMENT

  def Annotate(self, g: nx.MultiDiGraph, root_node: int) -> nx.MultiDiGraph:
    """Annotate nodes in the graph with dominator trees.

    The 'root node' annotation is a [0,1] value appended to node x vectors.
    The node label is a 1-hot binary vector set to y node vectors.

    Args:
      g: The graph to annotate.
      root_node: The root node for building the dominator tree.

    Returns:
      A data flow annotated graph.
    """
    function = g.nodes[root_node]["function"]

    if function is None:
      # Root node is outside of a function, so cannot dominate any other nodes.
      g.graph["data_flow_root_node"] = root_node
      g.graph["data_flow_steps"] = 0
      g.graph["data_flow_positive_node_count"] = 0
      return g

    # Because a node may only be dominated by a node from within the same
    # function, we need only consider the statements nodes within the same
    # function as the root node.
    statement_nodes: Set[int] = {
      node
      for node, data in g.nodes(data=True)
      if data["type"] == programl_pb2.Node.STATEMENT
      and data["function"] == function
    }
    # Pre-compute a mapping from statement to statement predecessors for all
    # statements in the function.
    predecessors: Dict[int, Set[int]] = {
      node: set(
        src
        for src, _, flow in g.in_edges(node, data="flow")
        if flow == programl_pb2.Edge.CONTROL
      )
      for node in statement_nodes
    }

    # Initialize the dominator sets. These map nodes to the set of nodes that
    # dominate it.
    initial_dominators = statement_nodes - set([root_node])
    dominators: Dict[int, Set[int]] = {
      n: initial_dominators for n in statement_nodes
    }
    dominators[root_node] = set([root_node])

    changed = True
    data_flow_steps = 0
    while changed:
      changed = False
      data_flow_steps += 1
      for node in dominators:
        if node == root_node:
          continue
        pred = predecessors[node]
        dom_pred = [dominators[p] for p in pred]
        if dom_pred:
          dom_pred = set.intersection(*dom_pred)
        else:
          dom_pred = set()
        new_dom = {node}.union(dom_pred)
        if new_dom != dominators[node]:
          dominators[node] = new_dom
          changed = True

    # Now that we have computed the dominator sets, assign labels and features
    # to all nodes.
    dominated_node_count = 0
    for node, data in g.nodes(data=True):
      data["x"].append(data_flow_graphs.ROOT_NODE_NO)
      if node in dominators and root_node in dominators[node]:
        dominated_node_count += 1
        data["y"] = DOMINATED
      else:
        data["y"] = NOT_DOMINATED

    g.nodes[root_node]["x"][-1] = data_flow_graphs.ROOT_NODE_YES

    g.graph["data_flow_root_node"] = root_node
    g.graph["data_flow_steps"] = data_flow_steps
    g.graph["data_flow_positive_node_count"] = dominated_node_count

    return g
