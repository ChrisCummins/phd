"""Module for labelling program graphs with data dependencies."""
import collections
from typing import List
from typing import Tuple

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_boolean(
  "only_entry_blocks_for_liveness_root_nodes",
  False,
  "Use only program entry nodes as the root for creating liveness graphs.",
)


# The node label arrays:
NOT_LIVE_OUT = [1, 0]
LIVE_OUT = [0, 1]


def GetDefsAndSuccessors(
  g: nx.MultiDiGraph, node: int
) -> Tuple[List[int], List[int]]:
  """Get the list of data elements that this statement defines, and the control
  successor statements."""
  defs, successors = [], []
  for src, dst, flow in g.out_edges(node, data="flow"):
    if flow == programl_pb2.Edge.DATA:
      defs.append(dst)
    elif flow == programl_pb2.Edge.CONTROL:
      successors.append(dst)
  return defs, successors


def GetUsesAndPredecessors(
  g: nx.MultiDiGraph, node: int
) -> Tuple[List[int], List[int]]:
  """Get the list of data elements which this statement uses, and control
  predecessor statements."""
  uses, predecessors = [], []
  for src, dst, flow in g.in_edges(node, data="flow"):
    if flow == programl_pb2.Edge.DATA:
      uses.append(src)
    elif flow == programl_pb2.Edge.CONTROL:
      predecessors.append(src)
  return uses, predecessors


def IsExitStatement(g: nx.MultiDiGraph, node: int):
  """Determine if a statement is an exit node."""
  for _, _, flow in g.out_edges(node, data="flow"):
    if flow == programl_pb2.Edge.CONTROL:
      break
  else:
    return True


class LivenessAnnotator(data_flow_graphs.NetworkXDataFlowGraphAnnotator):
  """Annotate graphs with liveness."""

  def __init__(self, *args, **kwargs):
    super(LivenessAnnotator, self).__init__(*args, **kwargs)
    # Liveness analysis begins at the exit block and works backwards.
    self.exit_nodes = [
      node
      for node, type_ in self.g.nodes(data="type")
      if type_ == programl_pb2.Node.STATEMENT and IsExitStatement(self.g, node)
    ]

    # Since we can't guarantee that input graphs have a single exit point, add
    # a temporary exit block which we will remove after computing liveness
    # results.
    liveness_start_node = self.g.number_of_nodes()
    assert liveness_start_node not in self.g.nodes
    self.g.add_node(liveness_start_node, type=programl_pb2.Node.STATEMENT)
    for exit_node in self.exit_nodes:
      self.g.add_edge(
        exit_node, liveness_start_node, flow=programl_pb2.Edge.CONTROL
      )

    # Ignore the liveness starting block when totalling up the data flow steps.
    data_flow_steps = -1

    # Create live-in and live-out maps that will be lazily evaluated.
    self.in_sets = collections.defaultdict(set)
    self.out_sets = collections.defaultdict(set)

    work_list = collections.deque([liveness_start_node])
    while work_list:
      data_flow_steps += 1
      node = work_list.popleft()
      defs, successors = GetDefsAndSuccessors(self.g, node)
      uses, predecessors = GetUsesAndPredecessors(self.g, node)

      # LiveOut(n) = U {LiveIn(p) for p in succ(n)}
      new_out_set = set().union(*[self.in_sets[p] for p in successors])

      # LiveIn(n) = Gen(n) U {LiveOut(n) - Kill(n)}
      new_in_set = set(uses).union(new_out_set - set(defs))

      # No need to visit predecessors if the in-set is non-empty and has not
      # changed.
      if not new_in_set or new_in_set != self.in_sets[node]:
        work_list.extend([p for p in predecessors if p not in work_list])

      self.in_sets[node] = new_in_set
      self.out_sets[node] = new_out_set

    # Remove the temporary node that we added.
    self.g.remove_node(liveness_start_node)

    self.data_flow_steps = data_flow_steps

  def IsValidRootNode(self, node: int, data) -> bool:
    """Liveness is a statement-based analysis."""
    return data["type"] == programl_pb2.Node.STATEMENT

  def Annotate(self, g: nx.MultiDiGraph, root_node: int) -> None:
    """Annotate nodes in the graph with liveness."""

    # A graph may not have any exit blocks.
    if not self.exit_nodes:
      g.graph["data_flow_root_node"] = root_node
      g.graph["data_flow_steps"] = 0
      g.graph["data_flow_positive_node_count"] = 0
      return

    # We have already pre-computed the live-out sets, so just add the
    # annotations.
    for _, data in g.nodes(data=True):
      data["x"].append(data_flow_graphs.ROOT_NODE_NO)
      data["y"] = NOT_LIVE_OUT
    g.nodes[root_node]["x"][-1] = data_flow_graphs.ROOT_NODE_YES

    for node in self.out_sets[root_node]:
      g.nodes[node]["y"] = LIVE_OUT

    g.graph["data_flow_root_node"] = root_node
    g.graph["data_flow_steps"] = self.data_flow_steps
    g.graph["data_flow_positive_node_count"] = len(self.out_sets[root_node])
