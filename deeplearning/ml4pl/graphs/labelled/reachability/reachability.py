"""Library for labelling program graphs with reachability information."""
import collections
import typing

import networkx as nx

from deeplearning.ml4pl.graphs import graph_query as query
from labm8.py import app
from labm8.py import decorators

FLAGS = app.FLAGS


@decorators.timeout(seconds=60)
def SetReachableNodes(
  g: nx.MultiDiGraph,
  root_node: str,
  x_label: str = "x",
  y_label: str = "y",
  true=True,
  false=False,
) -> typing.Tuple[int, int]:
  """Annotate nodes in the graph with x, y values for reachability.

  Args:
    g: The graph to annotate.
    root_node: The source node for determining reachability.

  Returns:
    The number of reachable nodes in the range 0 < n <= node_count, and the
    number of steps required to compute reachability for this graph.
  """
  # Initialize all nodes as unreachable and not root node, except the root node.
  for node, data in g.nodes(data=True):
    data[x_label] = [data[x_label], 0]
    data[y_label] = false
  g.nodes[root_node][x_label] = [g.nodes[root_node][x_label][0], 1]

  # Breadth-first traversal to mark reachable nodes.
  data_flow_steps = 0
  reachable_nodes_count = 0
  visited = set()
  q = collections.deque([(root_node, 1)])
  while q:
    next, data_flow_steps = q.popleft()
    reachable_nodes_count += 1
    visited.add(next)

    # Mark the node as reachable.
    g.nodes[next][y_label] = true

    for _, succ, flow in g.out_edges(next, data="flow"):
      if flow == "control" and succ not in visited:
        q.append((succ, data_flow_steps + 1))

  return reachable_nodes_count, data_flow_steps


def MakeReachabilityGraphs(
  g: nx.MultiDiGraph, n: typing.Optional[int] = None, false=False, true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` reachability graphs from the given unlabelled graph.

  Args:
    g: The unlabelled input graph.
    n: The maximum number of graphs to produce. Multiple graphs are produced by
      selecting different root nodes for creating reachability labels. If `n` is
      provided, the number of graphs generated will be in the range
      1 <= x <= min(num_statements, n). Else, the number og graphs will be equal
      to num_statements (i.e. one graph for each statement in the input graph).
    false: The value to set for binary false values on X and Y labels.
    true: The value to set for binary true values on X and Y labels.

  Returns:
    A generator of annotated graphs, where each graph has 'x' and 'y' labels on
    the statement nodes, and additionally 'reachable_node_count' and
    'data_flow_max_steps_required' attributes.
  """
  root_statements = query.SelectRandomNStatements(g, n)

  for root_node in root_statements[:n]:
    reachable = g.copy()
    (
      reachable.reachable_node_count,
      reachable.data_flow_max_steps_required,
    ) = SetReachableNodes(reachable, root_node, false=false, true=true)
    yield reachable
