"""Utility functions for graphs."""
import networkx as nx
import typing
import io

from labm8 import app
from deeplearning.ml4pl.graphs import graph_iterators as iterators

FLAGS = app.FLAGS

CreateLabelCallback = typing.Callable[[typing.Dict[str, typing.Any]], str]
KeyOrCallback = typing.Union[str, CreateLabelCallback]
StringOrCallback = typing.Union[str, CreateLabelCallback]


def GraphToDot(g: nx.Graph,
               statement_label: KeyOrCallback = 'text',
               statement_shape: StringOrCallback = 'box',
               statement_color: StringOrCallback = 'white',
               identifier_label: KeyOrCallback = 'name',
               identifier_shape: StringOrCallback = 'ellipse',
               identifier_color: StringOrCallback = 'white',
               magic_label: KeyOrCallback = 'name',
               magic_shape: StringOrCallback = 'doubleoctagon',
               magic_color: StringOrCallback = 'white',
               control_flow_color: StringOrCallback = 'blue',
               data_flow_label: KeyOrCallback = 'identifier',
               data_flow_color: StringOrCallback = 'red',
               call_edge_color: StringOrCallback = 'green'):
  """Render the dot visualiation of the graph.

  Args:
    node_label: The attribute to use as the node label.
    node_shape: The graphviz shape name to draw nodes in.
  """
  g = g.copy()

  def DataKeyOrCallback(data, key_or_callback) -> str:
    """Return key_or_callback(data) if callable, else data[key_or_callback]."""
    if callable(key_or_callback):
      return key_or_callback(data)
    else:
      return data.get(key_or_callback, '')

  def StringOrCallback(data, string_or_callback) -> str:
    """Return string_or_callback(data) if callable, else string_or_callback."""
    if callable(string_or_callback):
      return string_or_callback(data)
    else:
      return string_or_callback

  # Set node properties

  for _, data in g.nodes(data=True):
    # Add a 'null' attribute to nodes so that they can have empty labels.
    data['null'] = ''
    # Set the node to filled so that their color shows up.
    data['style'] = 'filled'

  for _, _, data in g.edges(data=True):
    # Add a 'null' attribute to edges so that they can have empty labels.
    data['null'] = ''

  for node, data in StatementNodeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, statement_label)}"'
    data['shape'] = StringOrCallback(data, statement_shape)
    data['fillcolor'] = StringOrCallback(data, statement_color)

  for node, data in IdentifierNodeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, identifier_label)}"'
    data['shape'] = StringOrCallback(data, identifier_shape)
    data['fillcolor'] = StringOrCallback(data, identifier_color)

  for node, data in MagicNodeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, magic_label)}"'
    data['shape'] = StringOrCallback(data, magic_shape)
    data['fillcolor'] = StringOrCallback(data, magic_color)

  # Set edge properties.

  for src, dst, data in iterators.ControlFlowEdgeIterator(g):
    data['color'] = StringOrCallback(data, control_flow_color)

  for src, dst, data in iterators.DataFlowEdgeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, data_flow_label)}"'
    data['color'] = StringOrCallback(data, data_flow_color)

  for src, dst, data in iterators.CallEdgeIterator(g):
    data['color'] = StringOrCallback(data, call_edge_color)

  # Remove unneeded attributes.
  def DeleteKeys(d, keys):
    for key in keys:
      if key in d:
        del d[key]

  for _, data in g.nodes(data=True):
    DeleteKeys(data, {'original_text', 'type', 'null', 'text', 'name'})
  for _, _, data in g.edges(data=True):
    DeleteKeys(data, {'flow', 'key'})

  buf = io.StringIO()
  nx.drawing.nx_pydot.write_dot(g, buf)
  return buf.getvalue()
