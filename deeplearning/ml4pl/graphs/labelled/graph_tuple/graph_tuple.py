"""The module implements conversion of graphs to tuples of arrays."""
import typing

import networkx as nx
import numpy as np
from labm8 import app

app.DEFINE_boolean(
    "tie_forward_and_backward_edge_types", False,
    "If true, insert backward edges using the same type as the forward edges. "
    "By default, backward edges are inserted using a different type")

FLAGS = app.FLAGS

# A mapping from a node to the number of incoming edges.
IncomingEdgeCount = typing.Dict[int, int]

# Perform the mapping between 'flow' property of edges and an index into a
# list of adjacency lists.
#
# TODO(cec): Replace the string constants with an enum for flow types.
FLOW_TO_EDGE_INDEX = {'control': 0, 'data': 1, 'call': 2}
# Perform the reverse maddping from index into adjacency list to 'flow'
# property.
EDGE_INDEX_TO_FLOW = {v: k for k, v in FLOW_TO_EDGE_INDEX.items()}
# Add lookup entries for backward edges.
for k, v in FLOW_TO_EDGE_INDEX.items():
  EDGE_INDEX_TO_FLOW[v + len(FLOW_TO_EDGE_INDEX)] = f'backward_{k}'


class GraphTuple(typing.NamedTuple):
  """The graph tuple: a compact, serializable representation of a graph."""

  # A list of adjacency lists, one for each edge_type, where an entry in an
  # adjacency list is a <src,dst> tuple of node indices.
  adjacency_lists: np.array  # Shape [edge_type_count, ?, 2], dtype int32

  # A list of edge positions, one for each edge type. An edge position is an
  # integer in the range 0 <= x < max_edge_position.
  edge_positions: np.array  # Shape [edge_type_count, ?], dtype int32

  # A list of incoming edge count dicts, one for each edge_type. Use
  # IncomingEdgeCountsToDense() to convert this to a dense representation.
  incoming_edge_counts: np.array  # Shape [edge_type_count, ?])

  # A list of indices into the node features table.
  node_x_indices: np.array  # Shape [node_count], dtype int32

  # (optional) A list of node arrays of node labels.
  # Shape [node_count, node_label_dimensionality]
  node_y: typing.Optional[np.array] = None

  # (optional) A list of indices into the graph features table.
  # Shape [graph_feature_dimensionality]
  graph_x: typing.Optional[np.array] = None

  # (optional) A vector of graph labels.
  graph_y: typing.Optional[
      np.array] = None  # Shape [graph_label_dimensionality]

  @property
  def has_node_y(self) -> bool:
    """Return whether graph tuple has node labels."""
    return self.node_y is not None

  @property
  def has_graph_x(self) -> bool:
    """Return whether graph tuple has graph features."""
    return self.graph_x is not None

  @property
  def has_graph_y(self) -> bool:
    """Return whether graph tuple has graph labels."""
    return self.graph_y is not None

  @property
  def node_count(self) -> int:
    """Return the number of nodes in the graph."""
    return len(self.node_x_indices)

  @property
  def edge_type_count(self) -> int:
    """Return the number of edge types."""
    return len(self.adjacency_lists)

  @property
  def dense_incoming_edge_counts(self) -> np.array:
    """Return counters for incoming edges as a dense array."""
    dense = np.zeros((self.node_count, self.edge_type_count))
    for edge_type, incoming_edge_dict in enumerate(self.incoming_edge_counts):
      for node_id, edge_count in incoming_edge_dict.items():
        dense[node_id, edge_type] = edge_count
    return dense

  @classmethod
  def CreateFromNetworkX(cls,
                         g: nx.MultiDiGraph,
                         node_y: str = 'y',
                         graph_x: str = 'x',
                         graph_y: str = 'y') -> 'GraphTuple':
    """Produce a tuple representation of a (multi-)directed networkx.

    Each edge must have a position and flow, and each node must have an 'x'
    embedding index.

    Args:
      g: The graph to convert to a graph_tuple.
      node_y: The property in node data dicts which stores the node label
        vector. If not present, node labels are omitted.
      graph_x: The property of the graph which stores the graph feature vector.
        If not present, graph features are omitted.
      graph_y: The property of the graph which stores the graph label vector.
        If not present, graph labels are omitted.

    Returns:
      A GraphTuple tuple.
    """
    edge_type_count = len(EDGE_INDEX_TO_FLOW)

    if not FLAGS.tie_forward_and_backward_edge_types:
      # Backward edges are inserted using a different type.
      edge_type_count *= 2

    # Create an adjacency list for each edge type.
    adjacency_lists: typing.List[typing.List[typing.Tuple[int, int]]] = [
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

    for src, dst, data in g.edges(data=True):
      flow = data['flow']
      position = data.get('position', 0)

      src_idx = node_to_index[src]
      dst_idx = node_to_index[dst]

      forward_edge_type = FLOW_TO_EDGE_INDEX[flow]
      backward_edge_type = GetBackwardEdgeType(forward_edge_type,
                                               edge_type_count)

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

    # Set node embedding indices.
    node_x_indices = [None] * g.number_of_nodes()
    for node, embedding_index in g.nodes(data='x'):
      if embedding_index is None:
        raise ValueError(f"No embedding for node `{node}`")
      node_idx = node_to_index[node]
      node_x_indices[node_idx] = embedding_index
    node_x_indices = np.array(node_x_indices, dtype=np.int32)

    # Set optional node labels.
    node_targets = [None] * g.number_of_nodes()
    for node, y in g.nodes(data=node_y, default=None):
      if y is None:
        node_y = None
        break
      node_idx = node_to_index[node]
      node_targets[node_idx] = y
    else:
      node_y = np.vstack(node_targets)

    # Set optional graph features.
    if graph_x and hasattr(g, graph_x):
      graph_x = np.array(getattr(g, graph_x, None))
    else:
      graph_x = None

    # Set optional graph label.
    if graph_y and hasattr(g, graph_y):
      graph_y = np.array(getattr(g, graph_y, None))
    else:
      graph_y = None

    return GraphTuple(
        adjacency_lists=adjacency_lists,
        edge_positions=edge_positions,
        incoming_edge_counts=incoming_edge_counts,
        node_x_indices=node_x_indices,
        node_y=node_y,
        graph_x=graph_x,
        graph_y=graph_y,
    )

  def ToNetworkx(self) -> nx.MultiDiGraph:
    """Construct a networkx graph from a graph tuple.

    Use this function for producing interpretable representation of graph
    tuples, but note that this is not an inverse of the CreateFromNetworkX()
    function, since critical information is lost, e.g. the mapping from edge
    flow types to their names, the name of feature and label attributes, etc.
    """
    g = nx.MultiDiGraph()

    for edge_type, (adjacency_list, position_list) in enumerate(
        zip(self.adjacency_lists, self.edge_positions)):
      for (src, dst), position in zip(adjacency_list, position_list):
        g.add_edge(src,
                   dst,
                   flow=EDGE_INDEX_TO_FLOW[edge_type],
                   position=position)

    for i, x in enumerate(self.node_x_indices):
      g.nodes[i]['x'] = x

    if self.has_node_y:
      for i, y in enumerate(self.node_y):
        g.nodes[i]['y'] = y

    if self.has_graph_x:
      g.x = self.graph_x

    if self.has_graph_y:
      g.y = self.graph_y

    return g


def GetBackwardEdgeType(forward_edge_type: int, edge_type_count: int):
  """Return the backward edge index for the given forward edge."""
  if FLAGS.tie_forward_and_backward_edge_types:
    return forward_edge_type
  else:
    return (edge_type_count // 2) + forward_edge_type
