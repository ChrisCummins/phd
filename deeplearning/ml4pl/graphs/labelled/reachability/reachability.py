"""Library for labelling program graphs with reachability information."""
import collections
import networkx as nx
import typing

from deeplearning.ml4pl.graphs import graph_query as query
from labm8 import app
from labm8 import decorators


FLAGS = app.FLAGS


@decorators.timeout(seconds=60)
def SetReachableNodes(g: nx.MultiDiGraph,
                      root_node: str,
                      max_steps: typing.Optional[int] = None,
                      x_label: str = 'x',
                      y_label: str = 'y',
                      true=True,
                      false=False) -> int:
  """Annotate nodes in the graph with x, y values for reachability.

  Args:
    g: The graph to annotate.
    root_node: The source node for determining reachability.
    max_steps: If > 0, limit the maximum number of steps permitted when
      computing reachability to this value.

  Returns:
    The true number of steps required to compute reachability for this graph.
    If max_steps > 0, this value is in the range 0 < n <= max_steps.
  """
  # Initialize all nodes as unreachable and not root node, except the root node.
  for node, data in g.nodes(data=True):
    data[x_label] = false
    data[y_label] = false
  g.nodes[root_node][x_label] = true

  # Breadth-first traversal to mark reachable nodes.
  steps = 0
  visited = set()
  q = collections.deque([(root_node, 0)])
  while q:
    next, steps = q.popleft()
    visited.add(next)
    if not max_steps or steps + 1 <= max_steps:
      for neighbor in query.StatementNeighbors(g, next):
        if neighbor not in visited:
          q.append((neighbor, steps + 1))

    # Mark the node as reachable.
    g.nodes[next][y_label] = true

  return steps
