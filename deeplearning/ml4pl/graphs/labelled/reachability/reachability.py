"""Library for labelling program graphs with reachability information."""
import collections
import networkx as nx
import random
import typing

from deeplearning.ml4pl.graphs import graph_iterators as iterators
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
                      false=False) -> typing.Tuple[int, int]:
  """Annotate nodes in the graph with x, y values for reachability.

  Args:
    g: The graph to annotate.
    root_node: The source node for determining reachability.
    max_steps: If > 0, limit the maximum number of steps permitted when
      computing reachability to this value.

  Returns:
    The number of reachable nodes in the range 0 < n <= node_count, and the
    number of steps required to compute reachability for this graph. If
    max_steps > 0, this value is in the range 0 < n <= max_steps.
  """
  # Initialize all nodes as unreachable and not root node, except the root node.
  for node, data in g.nodes(data=True):
    data[x_label] = false
    data[y_label] = false
  g.nodes[root_node][x_label] = true

  # Breadth-first traversal to mark reachable nodes.
  data_flow_steps = 0
  reachable_nodes_count = 0
  visited = set()
  q = collections.deque([(root_node, 1)])
  while q:
    next, data_flow_steps = q.popleft()
    reachable_nodes_count += 1
    visited.add(next)
    if not max_steps or data_flow_steps + 1 <= max_steps:
      for neighbor in query.StatementNeighbors(g, next):
        if neighbor not in visited:
          q.append((neighbor, data_flow_steps + 1))

    # Mark the node as reachable.
    g.nodes[next][y_label] = true

  return reachable_nodes_count, data_flow_steps


def MakeReachabilityGraphs(
    g: nx.MultiDiGraph,
    n: typing.Optional[int] = None,
    false=False,
    true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` reachability graphs

  :param g:
  :param n:
  :param false:
  :param true:
  :return:
  """
  nodes = [node for node, _ in iterators.StatementNodeIterator(g)]
  n = n or len(nodes)

  # If we're taking a sample of nodes to produce graphs (i.e. not all of them),
  # process the nodes in a random order.
  if n < len(nodes):
    random.shuffle(nodes)

  for node in nodes[:n]:
    reachable = g.copy()
    reachable.reachable_node_count, reachable.data_flow_max_steps_required = (
        SetReachableNodes(reachable,
                          node,
                          FLAGS.reachability_num_steps,
                          false=false,
                          true=true))
    yield reachable
