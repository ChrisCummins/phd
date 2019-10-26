"""Functions for iterating and maneuvering around graphs."""
import networkx as nx
from labm8 import app

FLAGS = app.FLAGS


def NodeTypeIterator(g: nx.DiGraph, type_name: str):
  """Iterate over nodes in a graph of a given type."""
  for node, data in g.nodes(data=True):
    # Default node type is statement.
    if data.get('type', 'statement') == type_name:
      yield node, data


def StatementNodeIterator(g: nx.DiGraph):
  """Iterate over the statement nodes in a graph."""
  yield from NodeTypeIterator(g, 'statement')


def IdentifierNodeIterator(g: nx.DiGraph):
  """Iterate over identifier nodes in a graph."""
  yield from NodeTypeIterator(g, 'identifier')


def MagicNodeIterator(g: nx.DiGraph):
  """Iterate over the statement nodes in a graph."""
  yield from NodeTypeIterator(g, 'magic')


def EdgeTypeIterator(g: nx.DiGraph, flow: str):
  """Iterate over edges of the given flow type."""
  for src, dst, data in g.edges(data=True):
    if data.get('flow', 'control') == flow:
      yield src, dst, data


def ControlFlowEdgeIterator(g: nx.DiGraph):
  """Iterate over the control flow edges of a graph."""
  yield from EdgeTypeIterator(g, 'control')


def DataFlowEdgeIterator(g: nx.DiGraph):
  """Iterate over the data flow edges of a graph."""
  yield from EdgeTypeIterator(g, 'data')


def CallEdgeIterator(g: nx.DiGraph):
  """Iterate over the call edges of a graph."""
  yield from EdgeTypeIterator(g, 'call')


def EntryBlockIterator(g: nx.Graph):
  """Iterate over the entry blocks of a graph.

  An entry block is a statement with no control predecessor.
  """
  for node, data in StatementNodeIterator(g):
    for src, dst, flow in g.in_edges(node, data='flow', default='control'):
      if flow == 'control':
        break
    else:
      yield node, data


def ExitBlockIterator(g: nx.Graph):
  """Iterate over the exit blocks of a graph.

  An exit block is a statement with no control successor.
  """
  for node, data in StatementNodeIterator(g):
    for src, dst, flow in g.out_edges(node, data='flow', default='control'):
      if flow == 'control':
        break
    else:
      yield node, data
