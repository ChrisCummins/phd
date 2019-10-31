"""The module implements conversion of graphs to dictionaries."""
import typing

import networkx as nx
import numpy as np
from labm8 import app

app.DEFINE_boolean(
    "tie_forward_and_backward_edge_types", False,
    "If true, insert backward edges using the same type as the forward edges. "
    "By default, backward edges are inserted using a different type")

FLAGS = app.FLAGS

NodeIndex = int

Edge = typing.Tuple[NodeIndex, NodeIndex]

AdjacencyList = typing.List[Edge]

# A mapping from a node to the number of incoming edges.
IncomingEdgeCount = typing.Dict[NodeIndex, int]

# Type alias for a graph dict.
GraphDict = typing.Dict[str, typing.Union[typing.List[AdjacencyList], typing.
                                          List[IncomingEdgeCount], typing.
                                          List[np.ndarray],]]


def GetBackwardEdgeType(forward_edge_type: int, edge_type_count: int):
  """Return the backward edge index for the given forward edge."""
  if FLAGS.tie_forward_and_backward_edge_types:
    return forward_edge_type
  else:
    return (edge_type_count // 2) + forward_edge_type


def ToGraphDict(g: nx.MultiDiGraph,
                edge_types: typing.Iterable[str],
                node_x_indices: str = 'x',
                node_y: str = 'y',
                graph_x: str = 'x',
                graph_y: str = 'y') -> GraphDict:
  """Produce a dictionary representation of a graph.

  Args:
    g: The graph to convert to a graph_dict.
    edge_types: The set of edge type names used. Defaults to {'call', 'control',
      'data'}.
    node_x_indices: The property in node data dicts which stores the node 
      embedding index.
    node_y: The property in node data dicts which stores the node label vector.
    graph_x: The property of the graph which stores the graph feature vector.
    graph_y: The property of the graph which stores the graph label vector.

  The dictionary has the following entries:

    adjacency_lists (np.array, Shape [edge_type_count, ?, 2]): A list of adjacency
      lists, one for each edge_type, where an entry in an adjacency list is a
      <src,dst> tuple of node indices.
    edge_positions (np.array, Shape [edge_type_count, ?]): A list of edge
      positions, one for each edge type. An edge position is an integer in the
      range 0 <= x < max_edge_position.
    incoming_edge_counts (np.array, Shape [edge_type_count, ?]): A list of
      incoming edge count dicts, one for each edge_type.
    node_x_indices (np.array, Shape [node_count]): A list of indices into the
      node features table.
    node_y (np.array, Shape [node_count, node_label_dimensionality]): A list
      node_x, but for labels.
    graph_x (np.array, Shape [graph_feature_dimensionality]): A list of
      indices into the graph features table.
    graph_y (np.array, Shape [graph_label_dimensionality]): A vector of graph
      labels.
  """
  edge_type_count = len(edge_types)
  if not FLAGS.tie_forward_and_backward_edge_types:
    # Backward edges are inserted using a different type.
    edge_type_count *= 2

  # Create an adjacency list for each edge type.
  adjacency_lists: typing.List[AdjacencyList] = [
      [] for _ in range(edge_type_count)
  ]
  # Create an edge position list for each edge type.
  edge_positions: typing.List[typing.List[int]] = [
      [] for _ in range(edge_type_count)
  ]
  # Lists of incoming edge counts for each mode, one for each edge type.
  incoming_edge_counts: typing.List[IncomingEdgeCount] = np.array(
      [{} for _ in range(edge_type_count)])

  # Create a mapping from node ID to a numeric ID.
  node_to_index = {node: i for i, node in enumerate(g.nodes)}
  # A mapping from edge 'flow' attribute to an index into the list of adjacency
  # lists.
  edge_type_to_index = {flow: i for i, flow in enumerate(sorted(edge_types))}

  for src, dst, data in g.edges(data=True):
    flow = data['flow']
    position = data['position']

    src_idx = node_to_index[src]
    dst_idx = node_to_index[dst]

    forward_edge_type = edge_type_to_index[flow]
    backward_edge_type = GetBackwardEdgeType(forward_edge_type, edge_type_count)

    # Add the forward and backward edges.
    forward_adjacency_list = adjacency_lists[forward_edge_type]
    backward_adjacency_list = adjacency_lists[backward_edge_type]
    forward_adjacency_list.append((src_idx, dst_idx))
    backward_adjacency_list.append((dst_idx, src_idx))

    # Add the edge positions.
    forward_position_list = edge_positions[forward_edge_type]
    backward_position_list = edge_positions[backward_edge_type]
    forward_position_list.append(position)
    backward_position_list.append(position)

    # Update the incoming edge counts.
    incoming_edge_count_dict = incoming_edge_counts[forward_edge_type]
    incoming_edge_count_dict[dst_idx] = (
        incoming_edge_count_dict.get(dst_idx, 0) + 1)
    incoming_edge_count_dict = incoming_edge_counts[backward_edge_type]
    incoming_edge_count_dict[src_idx] = (
        incoming_edge_count_dict.get(src_idx, 0) + 1)

  # Convert to numpy arrays.
  adjacency_lists = np.array([
      np.array(adjacency_list, dtype=np.int32)
      for adjacency_list in adjacency_lists
  ])
  edge_positions = np.array([
      np.array(edge_position, dtype=np.int32)
      for edge_position in edge_positions
  ])

  graph_dict = {
      'adjacency_lists': adjacency_lists,
      'edge_positions': edge_positions,
      'incoming_edge_counts': incoming_edge_counts,
  }

  # Set node embedding indices.
  if node_x_indices and node_x_indices in g.nodes[src]:
    node_embedding_indices = [None] * g.number_of_nodes()
    for node, embedding_index in g.nodes(data=node_x_indices):
      node_idx = node_to_index[node]
      node_embedding_indices[node_idx] = embedding_index
    graph_dict['node_x_indices'] = np.array(node_embedding_indices,
                                            dtype=np.int32)

  # Set node labels.
  if node_y and node_y in g.nodes[src]:
    node_targets = [None] * g.number_of_nodes()
    for node, y in g.nodes(data=node_y):
      node_idx = node_to_index[node]
      node_targets[node_idx] = y
    graph_dict['node_y'] = np.vstack(node_targets)

  # Set graph features.
  if graph_x and hasattr(g, graph_x):
    graph_dict['graph_x'] = np.array(getattr(g, graph_x, None))

  # Set graph label.
  if graph_y and hasattr(g, graph_y):
    graph_dict['graph_y'] = np.array(getattr(g, graph_y, None))

  return graph_dict


def IncomingEdgeCountsToDense(
    incoming_edge_counts: typing.List[IncomingEdgeCount], node_count: int,
    edge_type_count: int) -> np.ndarray:
  """Turn counters for incoming edges into a dense array."""
  dense = np.zeros((node_count, edge_type_count))
  for edge_type, incoming_edge_dict in enumerate(incoming_edge_counts):
    for node_id, edge_count in incoming_edge_dict.items():
      dense[node_id, edge_type] = edge_count
  return dense


def GraphDictToNetworkx(graph_dict: GraphDict) -> nx.MultiDiGraph:
  """Construct a networkx graph from a graph dict.

  Use this function for producing interpretable representation of graph_dicts,
  but note that this is not an inverse of the ToGraphDict() function, since
  critical information is lost, e.g. the mapping from edge flow types to numeric
  ID. the name of feature and label attributes, etc.
  """
  g = nx.MultiDiGraph()

  # Build the list of edges and their properties by iterating over the adjacency
  # lists and producing a flat list of edge dicts which can then be augmented
  # with features or labels.
  flattened_edge_dicts: typing.List[typing.Dict[str, typing.Any]] = []

  for edge_type, adjacency_list in enumerate(graph_dict['adjacency_lists']):
    for src, dst in adjacency_list:
      flattened_edge_dicts.append({'src': src, 'dst': dst, 'flow': edge_type})

  # Add the edges and their properties to the graph.
  for edge_dict in flattened_edge_dicts:
    src, dst = edge_dict['src'], edge_dict['dst']
    del edge_dict['src']
    del edge_dict['dst']
    g.add_edge(src, dst, **edge_dict)

  if 'node_x_indices' in graph_dict:
    for i, x in enumerate(graph_dict['node_x_indices']):
      g.nodes[i]['x'] = x

  if 'node_y' in graph_dict:
    for i, y in enumerate(graph_dict['node_y']):
      g.nodes[i]['y'] = y

  if 'graph_x' in graph_dict:
    g.x = graph_dict['graph_x']

  if 'graph_y' in graph_dict:
    g.y = graph_dict['graph_y']

  return g
