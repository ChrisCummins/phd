"""Utility functions for graphs."""
import io
import networkx as nx
import pathlib
import tempfile
import typing
import zipfile

from labm8 import app
from labm8 import fs
from labm8 import labtypes


FLAGS = app.FLAGS

CreateLabelCallback = typing.Callable[[typing.Dict[str, typing.Any]], str]
KeyOrCallback = typing.Union[str, CreateLabelCallback]
StringOrCallback = typing.Union[str, CreateLabelCallback]


def _NodeLabel(data):
  """Helper callback to set default node labels."""
  node_type = data.get('type', 'statement')
  if node_type == 'statement':
    return data.get('text', '')
  elif node_type == 'identifier':
    return data.get('name', '')
  elif node_type == 'magic':
    return data.get('name', '')
  else:
    return ''


def _NodeShape(data):
  """Helper callback to set default node shapes."""
  node_type = data.get('type', 'statement')
  if node_type == 'statement':
    return 'box'
  elif node_type == 'identifier':
    return 'ellipse'
  elif node_type == 'magic':
    return 'doubleoctagon'
  else:
    return ''


def _EdgeColor(data):
  """Helper callback to set default edge colors."""
  flow = data.get('flow')
  if flow == 'control':
    return 'blue'
  elif flow == 'data':
    return 'red'
  elif flow == 'call':
    return 'green'
  else:
    return 'black'


def GraphToDot(g: nx.Graph,
               node_label: KeyOrCallback = _NodeLabel,
               node_shape: StringOrCallback = _NodeShape,
               node_color: StringOrCallback = 'white',
               edge_label: KeyOrCallback = 'null',
               edge_color: StringOrCallback = _EdgeColor) -> str:
  """Render the dot visualization of the graph.

  Args:
    g: The graph to render.
    node_label: The key into a node's data dictionary to produce a label, or a
      callback that takes as input the data dictionary and returns a label.
    node_shape: The shape of a node, else a callback that takes as input the
      data dictionary and returns a shape.
    node_color: The color of a node, else a callback that takes as input the
      data dictionary and returns a color.
    edge_label: The key into an edges's data dictionary to produce a label, or a
      callback that accepts the data dictionary as an argument and returns a
      label.
    edge_color: The color of a edge, else a callback that takes as input the
      data dictionary and returns a color.

  Returns:
    A dot graph string.
  """
  g = g.copy()  # Non-destructive edits.

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

    data['label'] = f'"{DataKeyOrCallback(data, node_label)}"'
    data['shape'] = StringOrCallback(data, node_shape)
    # Set the node to filled so that their color shows up.
    data['style'] = 'filled'
    data['fillcolor'] = StringOrCallback(data, node_color)

    # Remove unneeded attributes.
    labtypes.DeleteKeys(data, {'original_text', 'type', 'null', 'text', 'name'})

  for _, _, data in g.edges(data=True):
    # Add a 'null' attribute to edges so that they can have empty labels.
    data['null'] = ''

    data['label'] = f'"{DataKeyOrCallback(data, edge_label)}"'
    data['color'] = StringOrCallback(data, edge_color)

    # Remove unneeded attributes.
    labtypes.DeleteKeys(data, {'flow', 'key', 'null'})

  buf = io.StringIO()
  nx.drawing.nx_pydot.write_dot(g, buf)
  return buf.getvalue()


def InputOutputGraphsToDotZip(
    input_output_graphs: typing.List[typing.Tuple[nx.Graph, nx.Graph]],
    path: pathlib.Path,
    input_graph_to_dot: typing.Callable[[nx.Graph], str] = GraphToDot,
    output_graph_to_dot: typing.Callable[[nx.Graph], str] = GraphToDot
) -> pathlib.Path:
  """Write the given <input,output> graph tuples to a zip archive.

  Args:
    input_output_graphs: A list of <input,output> graph tuples.
    path: The path of the zip archive to create. If it already exists, it is
      overwritten.
    input_graph_to_dot: A callback to render the dot string for an input graph.
    output_graph_to_dot: A callback to render the dot string for an output
      graph.

  Returns:
    The path of the generated zip archive.
  """
  if path.is_file():
    path.unlink()

  with tempfile.TemporaryDirectory(prefix="phd_") as d:

    with fs.chdir(d), zipfile.ZipFile(path, 'w') as zip:
      for i, (input_graph, output_graph) in enumerate(input_output_graphs):
        input_graph_dot = f'graph_{i:04d}_input.dot'
        output_graph_dot = f'graph_{i:04d}_output.dot'

        fs.Write(input_graph_dot,
                 input_graph_to_dot(input_graph).encode('utf-8'))
        fs.Write(output_graph_dot,
                 output_graph_to_dot(output_graph).encode('utf-8'))

        zip.write(input_graph_dot)
        zip.write(output_graph_dot)

  return path
