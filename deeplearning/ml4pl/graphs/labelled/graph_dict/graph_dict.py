"""The module implements conversion of graphs to dictionaries."""
import networkx as nx
import numpy as np
import typing

from labm8 import app

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


def ToGraphDict(g: nx.MultiDiGraph,
                edge_types: typing.Iterable[str],
                node_x: str = 'x',
                node_y: str = 'y',
                edge_x: str = 'x',
                edge_y: str = 'y',
                graph_x: str = 'x',
                graph_y: str = 'y') -> GraphDict:
  """

  Produce a dictionary representation of a graph. The dictionary has the
  following properties:

    adjacency_lists (Shape [edge_type_count, ?]): A list of adjacency lists,
      one for each edge_type, where an entry in an adjacency list is a
      <src,dst> tuple of node indices.
    incoming_edge_counts (Shape [edge_type_count, ?]): A list of incoming edge
      count dicts.
    edge_x (Shape [edge_type_count, ?, edge_feature_dimensionality]): A matrix
      of edge features with the same shape as adjacency_lists, but instead of
      tuples, each item is a feature vector.
    edge_y (Shape [edge_type_count, ?, edge_label_dimensionality]): Same
      as edge_x, but for labels.
    node_x (Shape [node_count, node_feature_dimensionality]): A list of node
      feature vectors.
    node_y (Shape [node_count, node_label_dimensionality]): Same as node_x, but
      for labels.
    graph_x (Shape [graph_feature_dimensionality]): A vector of graph features.
    graph_y (Shape [graph_label_dimensionality]): A vector of graph labels.
  """
  # TODO(cec): This implementation only supports a single node type.

  # Create an adjacency list for each edge type.
  adjacency_lists: typing.List[AdjacencyList] = []
  # Lists of incoming edge counts for each mode, one for each edge type.
  incoming_edge_counts: typing.List[IncomingEdgeCount] = []

  # Initialize the per-edge type structures.
  for _ in range(len(edge_types)):
    adjacency_lists.append([])
    incoming_edge_counts.append({})

  # Create a mapping from node ID to a numeric ID.
  node_to_index = {node: i for i, node in enumerate(g.nodes)}
  # A mapping from edge 'flow' attribute to an index into the list of adjacency
  # lists.
  edge_type_to_index = {flow: i for i, flow in enumerate(sorted(edge_types))}

  for src, dst, flow in g.edges(data='flow'):
    src_idx = node_to_index[src]
    dst_idx = node_to_index[dst]
    edge_type_index = edge_type_to_index[flow]
    adjacency_list = adjacency_lists[edge_type_index]
    adjacency_list.append((src_idx, dst_idx))

    incoming_edge_count_dict = incoming_edge_counts[edge_type_index]
    incoming_edge_count_dict[dst_idx] = (
        incoming_edge_count_dict.get(dst_idx, 0) + 1)

  adjacency_lists = [
      np.array(adjacency_list, dtype=np.int32)
      for adjacency_list in adjacency_lists
  ]

  graph_dict = {
      'adjacency_lists': adjacency_lists,
      'incoming_edge_counts': incoming_edge_counts,
  }

  # Set node features.
  if node_x and node_x in g.nodes[src]:
    node_features = [None] * g.number_of_nodes()
    for node, x in g.nodes(data=node_x):
      node_idx = node_to_index[node]
      node_features[node_idx] = x
    graph_dict['node_x'] = node_features

  # Set node labels.
  if node_y and node_y in g.nodes[src]:
    node_targets = [None] * g.number_of_nodes()
    for node, y in g.nodes(data=node_y):
      node_idx = node_to_index[node]
      node_targets[node_idx] = y
    graph_dict['node_y'] = node_targets

  # Set edge features.
  if edge_x and edge_x in g.edges[src, dst, 0]:
    edge_features_lists = []
    for _ in range(len(edge_types)):
      edge_features_lists.append([])
    for src, dst, data in g.edges(data=True):
      edge_type_index = edge_type_to_index[data['flow']]
      edge_features = edge_features_lists[edge_type_index]
      edge_features.append(data[edge_x])

    graph_dict['edge_x'] = [
        np.array(edge_features, dtype=np.int32)
        for edge_features in edge_features_lists
    ]

  # Set edge labels.
  if edge_y and edge_y in g.edges[src, dst, 0]:
    edge_targets_lists = []
    for _ in range(len(edge_types)):
      edge_targets_lists.append([])
    for src, dst, data in g.edges(data=True):
      edge_type_index = edge_type_to_index[data['flow']]
      edge_targets = edge_targets_lists[edge_type_index]
      edge_targets.append(data[edge_y])

    graph_dict['edge_y'] = [
        np.array(edge_targets, dtype=np.int32)
        for edge_targets in edge_targets_lists
    ]

  # Set graph features.
  if graph_x and hasattr(g, graph_x):
    graph_dict['graph_x'] = getattr(g, graph_x, None)

  # Set graph label.
  if graph_y and hasattr(g, graph_y):
    graph_dict['graph_y'] = getattr(g, graph_y, None)

  return graph_dict


def IncomingEdgeCountsToDense(
    incoming_edge_counts: typing.List[IncomingEdgeCount], node_count: int,
    edge_type_count: int) -> np.ndarray:
  # Turn counters for incoming edges into a dense array:
  dense = np.zeros((node_count, edge_type_count))
  for edge_type, incoming_edge_dict in enumerate(incoming_edge_counts):
    for node_id, edge_count in incoming_edge_dict.items():
      dense[node_id, edge_type] = edge_count
  return dense
