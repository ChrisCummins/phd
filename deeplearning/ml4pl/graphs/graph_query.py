"""Functions for iterating and maneuvering around graphs."""
import collections
import networkx as nx
import random
import typing

from deeplearning.ml4pl.graphs import graph_iterators as iterators
from labm8 import app

FLAGS = app.FLAGS


def StatementNeighbors(g: nx.Graph, node: str,
                       flow='control') -> typing.Set[str]:
  """Return the neighboring statements connected by the given flow type."""
  neighbors = set()
  for src, dst in g.out_edges(node):
    if g.edges[src, dst, 0]['flow'] == flow:
      if g.nodes[dst].get('type', 'statement') == 'statement':
        neighbors.add(dst)
      else:
        neighbors = neighbors.union(StatementNeighbors(g, dst, flow=flow))
  return neighbors


def SuccessorNodes(
    g: nx.DiGraph,
    node: str,
    direction: typing.Optional[
        typing.Callable[[typing.Any, typing.Any], typing.Any]] = None,
    ignored_nodes: typing.Optional[typing.Iterable[str]] = None
) -> typing.List[str]:
  """Find the successor nodes of a node."""
  direction = direction or (lambda src, dst: dst)
  ignored_nodes = ignored_nodes or set()
  real = []
  for src, dst in direction(g.in_edges, g.out_edges)(node):
    if direction(src, dst) not in ignored_nodes:
      real.append(direction(src, dst))
    else:
      # The node is ignored, so skip over it and look for the next
      real += SuccessorNodes(g, direction(src, dst), ignored_nodes, direction)
  return real


def StatementIsSuccessor(g: nx.MultiDiGraph,
                         src: str,
                         dst: str,
                         flow: str = 'control') -> bool:
  """Return True if `dst` is successor to `src`."""
  visited = set()
  q = collections.deque([src])
  while q:
    current = q.popleft()
    if current == dst:
      return True
    visited.add(current)
    for next_node in g.neighbors(current):
      edge_flow = g.edges[current, next_node, 0].get('flow', 'control')
      if edge_flow != flow:
        continue
      node_type = g.nodes[next_node].get('type', 'statement')
      if node_type != 'statement':
        continue
      if next_node in visited:
        continue
      q.append(next_node)
  return False


def SelectRandomNStatements(g: nx.Graph, n: int):
  root_statements = [node for node, _ in iterators.StatementNodeIterator(g)]
  n = n or len(root_statements)

  # If we're taking a sample of nodes to produce graphs (i.e. not all of them),
  # process the nodes in a random order.
  if n < len(root_statements):
    random.shuffle(root_statements)

  return root_statements
