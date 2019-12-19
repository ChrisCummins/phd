# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for iterating over and querying networkx program graphs."""
import collections
import typing
from typing import List
from typing import Tuple

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app


FLAGS = app.FLAGS

###############################################################################
# Iterators.
###############################################################################


def NodeTypeIterator(g: nx.DiGraph, node_type: programl_pb2.Node.Type):
  """Iterate over nodes in a graph of a given type."""
  for node, data in g.nodes(data=True):
    if data["type"] == node_type:
      yield node, data


def StatementNodeIterator(g: nx.DiGraph):
  """Iterate over the statement nodes in a graph."""
  yield from NodeTypeIterator(g, programl_pb2.Node.STATEMENT)


def IdentifierNodeIterator(g: nx.DiGraph):
  """Iterate over identifier nodes in a graph."""
  yield from NodeTypeIterator(g, programl_pb2.Node.IDENTIFIER)


def EdgeTypeIterator(g: nx.DiGraph, flow: programl_pb2.Edge.Flow):
  """Iterate over edges of the given flow type."""
  for src, dst, data in g.edges(data=True):
    if data["flow"] == flow:
      yield src, dst, data


def ControlFlowEdgeIterator(g: nx.DiGraph):
  """Iterate over the control flow edges of a graph."""
  yield from EdgeTypeIterator(g, programl_pb2.Edge.CONTROL)


def DataFlowEdgeIterator(g: nx.DiGraph):
  """Iterate over the data flow edges of a graph."""
  yield from EdgeTypeIterator(g, programl_pb2.Edge.DATA)


def CallEdgeIterator(g: nx.DiGraph):
  """Iterate over the call edges of a graph."""
  yield from EdgeTypeIterator(g, programl_pb2.Edge.CALL)


def EntryBlockIterator(g: nx.DiGraph):
  """Iterate over the entry blocks of a graph.
  An entry block is a statement with no control predecessor.
  """
  for node, data in StatementNodeIterator(g):
    # Filter out the root node.
    if data["function"] is None:
      continue

    for _, _, flow in g.in_edges(node, data="flow"):
      if flow == programl_pb2.Edge.CONTROL:
        break
    else:
      yield node, data


def ExitBlockIterator(g: nx.Graph):
  """Iterate over the exit blocks of a graph.
  An exit block is a statement with no control successor.
  """
  for node, data in StatementNodeIterator(g):
    # Filter out the root node.
    if data["function"] is None:
      continue

    for _, _, flow in g.out_edges(node, data="flow"):
      if flow == programl_pb2.Edge.CONTROL:
        break
    else:
      yield node, data


###############################################################################
# Queries.
###############################################################################


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


def StatementIsSuccessor(
  g: nx.MultiDiGraph,
  src: int,
  dst: int,
  flow: programl_pb2.Edge.Flow = programl_pb2.Edge.CONTROL,
) -> bool:
  """Return True if `dst` is successor to `src`."""
  visited = set()
  q = collections.deque([src])
  while q:
    current = q.popleft()
    if current == dst:
      return True
    visited.add(current)
    for _, next_node, edge_flow in g.out_edges(current, data="flow"):
      if edge_flow != flow:
        continue
      node_type = g.nodes[next_node]["type"]
      if node_type != programl_pb2.Node.STATEMENT:
        continue
      if next_node in visited:
        continue
      q.append(next_node)
  return False


def StatementNeighbors(
  g: nx.Graph,
  node: int,
  flow=programl_pb2.Edge.CONTROL,
  direction: typing.Optional[
    typing.Callable[[typing.Any, typing.Any], typing.Any]
  ] = None,
) -> typing.Set[int]:
  """Return the neighboring statements connected by the given flow type."""
  direction = direction or (lambda src, dst: dst)
  neighbors = set()
  neighbor_edges = direction(g.in_edges, g.out_edges)
  for src, dst, edge_flow in neighbor_edges(node, data="flow"):
    neighbor = direction(src, dst)
    if edge_flow == flow:
      if g.nodes[neighbor]["type"] == programl_pb2.Node.STATEMENT:
        neighbors.add(neighbor)
      else:
        neighbors = neighbors.union(StatementNeighbors(g, neighbor, flow=flow))
  return neighbors


def SuccessorNodes(
  g: nx.DiGraph,
  node: int,
  direction: typing.Optional[
    typing.Callable[[typing.Any, typing.Any], typing.Any]
  ] = None,
  ignored_nodes: typing.Optional[typing.Iterable[int]] = None,
) -> typing.List[int]:
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


def GetStatementsForNode(
  graph: nx.MultiDiGraph, node: int
) -> typing.Iterable[str]:
  """Return the statements which are connected to the given node.

  Args:
    graph: The graph to fetch the statements from.
    node: The node to fetch the statements for. If the node is a statement, it
      returns itself. If the node is an identifier, it returns all statements
      which define/use this identifier.

  Returns:
    An iterator over statement nodes.
  """
  root = graph.nodes[node]
  if root["type"] == programl_pb2.Node.STATEMENT:
    yield node
  elif (
    root["type"] == programl_pb2.Node.IDENTIFIER
    or root["type"] == programl_pb2.Node.IMMEDIATE
  ):
    for src, dst in graph.in_edges(node):
      if graph.nodes[src]["type"] == programl_pb2.Node.STATEMENT:
        yield src
    for src, dst in graph.out_edges(node):
      if graph.nodes[dst]["type"] == programl_pb2.Node.STATEMENT:
        yield dst
  else:
    raise NotImplementedError(f"Unknown node type {root['type']}")
