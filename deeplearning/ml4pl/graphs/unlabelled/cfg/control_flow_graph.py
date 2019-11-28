"""A class representing a control flow graph."""
import typing

import networkx as nx

from deeplearning.ml4pl import ml4pl_pb2
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS


class MalformedControlFlowGraphError(ValueError):
  """Base class for errors raised by ValidateControlFlowGraph."""
  pass


class NotEnoughNodes(MalformedControlFlowGraphError):
  pass


class GraphContainsSelfLoops(MalformedControlFlowGraphError):
  pass


class UnconnectedNode(MalformedControlFlowGraphError):
  pass


class InvalidNodeDegree(MalformedControlFlowGraphError):
  pass


class MissingNodeName(MalformedControlFlowGraphError):
  pass


class DuplicateNodeName(MalformedControlFlowGraphError):
  pass


class InvalidSpecialBlock(MalformedControlFlowGraphError):
  pass


class NoEntryBlock(InvalidSpecialBlock):
  pass


class MultipleEntryBlocks(InvalidSpecialBlock):
  pass


class NoExitBlock(InvalidSpecialBlock):
  pass


class ControlFlowGraph(nx.DiGraph, pbutil.ProtoBackedMixin):
  """A control flow graph.

  For a control flow graph to be considered "valid", the following properties
  should be adhered to:

   * All nodes in the graph should have a unique "name" attribute. This can be
     set at creation time, e.g.: cfg.add_node(0, name='foo').
   * The graph should be fully connected.

  Use the IsValidControlFlowGraph() method to check if a graph instance has
  these properties.
  """

  proto_t = ml4pl_pb2.ControlFlowGraph

  def __init__(self, name: str = 'cfg'):
    super(ControlFlowGraph, self).__init__(name=name)

  def ToSuccessorsString(self) -> str:
    """Format graph as a sequence of node descendants lists."""
    ret = []
    nodes = sorted(self.nodes(data=True), key=lambda n: n[1]['name'])
    for index, node in nodes:
      descendants = [self.nodes[n]['name'] for n in nx.descendants(self, index)]
      ret.append(f"{node['name']}: {' '.join(sorted(descendants))}")
    return '\n'.join(ret)

  def ToNeighborsString(self) -> str:
    """Format graph as a sequence of node neighbor lists."""
    ret = []
    nodes = sorted(self.nodes(data=True), key=lambda n: n[1]['name'])
    for index, node in nodes:
      descendants = [self.nodes[n]['name'] for n in nx.neighbors(self, index)]
      ret.append(f"{node['name']}: {' '.join(sorted(descendants))}")
    return '\n'.join(ret)

  def ValidateControlFlowGraph(self, strict: bool = True) -> 'ControlFlowGraph':
    """Determine if the graph is a valid control flow graph.

    Args:
      strict: If True, check that the graph follows all properties of a CFG,
        i.e. that all nodes have the expected degrees of inputs and outputs, so
        that no edges can be fused.

    Returns:
      The graph, i.e. 'self'.

    Raises:
      MalformedControlFlowGraphError: If the graph is not a valid CFG. The
        exception will be some valid subclass of this base error, with an
        informative error message.
    """
    number_of_nodes = self.number_of_nodes()

    # CFGs must contain one or more nodes.
    if number_of_nodes < 1:
      raise NotEnoughNodes(f'Function `{self.name}` has no nodes')

    # Get the entry and exit blocks. These properties will raise exceptions
    # if they are not found / duplicates found.
    entry_node = self.entry_block
    exit_nodes = self.exit_blocks

    out_degrees = {n: self.out_degree(n) for n in self.nodes}
    in_degrees = {n: self.in_degree(n) for n in self.nodes}

    if number_of_nodes > 1:
      if entry_node in exit_nodes:
        raise InvalidSpecialBlock(f'Exit and entry nodes are the same: '
                                  f"'{self.nodes[entry_node]['name']}'")

      for exit_node in exit_nodes:
        if not nx.has_path(self, entry_node, exit_node):
          raise MalformedControlFlowGraphError(
              f"No path from entry node '{self.nodes[entry_node]['name']}' to "
              f"exit node '{self.nodes[exit_node]['name']}' in function "
              f"`{self.name}`")

    # Validate node attributes.
    node_names = set()
    for node in self.nodes:
      # All nodes must have a name.
      if 'name' not in self.nodes[node]:
        raise MissingNodeName(
            f'Node {node} has no name in function `{self.name}`')

      # All node names must be unique.
      node_name = self.nodes[node]['name']
      if node_name in node_names:
        raise DuplicateNodeName(
            f"Duplicate node name '{node_name}' in function `{self.name}`")
      node_names.add(node_name)

      # All nodes must be connected (except for 1-node graphs).
      if number_of_nodes > 1 and not out_degrees[node] + in_degrees[node]:
        raise UnconnectedNode(f"Unconnected node '{self.nodes[node]['name']}'")

    # The entry node has an additional input, since it must entered.
    in_degrees[entry_node] += 1

    # The exit block cannot have outputs.
    for exit_node in exit_nodes:
      if out_degrees[exit_node]:
        app.Error("OUT DEGREE %s", self.out_degree(exit_node))
        raise InvalidNodeDegree(
            f"Exit block outdegree({self.nodes[exit_node]['name']}) = "
            f'{out_degrees[exit_node]} in function `{self.name}`')

    # Additional "strict" CFG tests.
    if strict:
      # Validate edge attributes.
      for src, dst in self.edges:
        if src == dst:
          raise GraphContainsSelfLoops(f'Self loops: {src} -> {dst}')

        # Each node in a CFG must have more than one output, or more than one
        # input. This is because nodes represent basic blocks: a node with only
        # a single output should have been fused with the consuming node (i.e.
        # they are the same basic block).
        if not (out_degrees[src] > 1 or in_degrees[dst] > 1):
          raise InvalidNodeDegree(
              f"outdegree({self.nodes[src]['name']}) = {out_degrees[src]}, "
              f"indegree({self.nodes[dst]['name']}) = {in_degrees[dst]}")

    return self

  def IsValidControlFlowGraph(self, strict: bool = True) -> bool:
    """Return true if the graph is a valid control flow graph.

    Args:
      strict: If True, check that the graph follows all properties of a CFG,
        i.e. that all nodes have the expected degrees of inputs and outputs, so
        that no edges can be fused.
    """
    try:
      self.ValidateControlFlowGraph(strict=strict)
      return True
    except MalformedControlFlowGraphError:
      return False

  def IsEntryBlock(self, node) -> bool:
    """Return if the given node is an entry block."""
    return self.nodes[node].get('entry', False)

  def IsExitBlock(self, node) -> bool:
    """Return if the given node is an exit block."""
    return self.nodes[node].get('exit', False)

  @property
  def entry_block(self) -> int:
    """Return the entry block."""
    entry_blocks = [n for n in self.nodes if self.IsEntryBlock(n)]
    if not entry_blocks:
      raise NoEntryBlock()
    elif len(entry_blocks) > 1:
      raise MultipleEntryBlocks()
    return entry_blocks[0]

  @property
  def exit_blocks(self) -> typing.List[int]:
    """Return the exit blocks."""
    exit_blocks = [n for n in self.nodes if self.IsExitBlock(n)]
    if not exit_blocks:
      raise NoExitBlock()
    return exit_blocks

  @property
  def edge_density(self) -> float:
    """The edge density is the ratio of edges to fully connected, [0,1]."""
    return self.number_of_edges() / (self.number_of_nodes() *
                                     self.number_of_nodes())

  @property
  def undirected_diameter(self) -> int:
    """Get the diameter of the CFG as an undirected graph.

    The diameter of the graph is the maximum eccentricity, where eccentricity
    is the maximum distance from one node to all other nodes.

    A CFG can never be strongly connected, since the exit node always has an
    outdegree of 0.
    """
    return nx.diameter(self.to_undirected())

  def SetProto(self, proto: pbutil.ProtocolBuffer) -> None:
    # Ensure that graph is valid. This will raise exception if graph is not
    # valid.
    self.ValidateControlFlowGraph(strict=False)

    # Set the graph-level properties.
    proto.name = self.graph['name']
    proto.entry_block_index = self.entry_block
    proto.exit_block_index[:] = self.exit_blocks

    # We translate node IDs to indexes into the node list.
    node_to_index: typing.Dict[int, int] = {}

    # Create the block protos.
    for i, (node, data) in enumerate(self.nodes(data=True)):
      node_to_index[node] = i
      block = proto.block.add()
      block.name = data['name']
      text = data.get('text')
      if text:
        block.text = data.get('text')
    # Create the edge protos.
    for src, dst in self.edges:
      edge = proto.edge.add()
      edge.src_index = node_to_index[src]
      edge.dst_index = node_to_index[dst]

  @classmethod
  def FromProto(cls, proto: pbutil.ProtocolBuffer) -> 'ProtoBackedMixin':
    instance = cls(name=proto.name)
    # Create the nodes from the block protos.
    for i, block in enumerate(proto.block):
      data = {'name': block.name}
      if block.text:
        data['text'] = block.text
      instance.add_node(i, **data)
    # Set the special block attributes.
    instance.nodes[proto.entry_block_index]['entry'] = True
    try:
      for index in proto.exit_block_index:
        instance.nodes[index]['exit'] = True
    except TypeError:
      # I changed the exit_block_index field from singular to repeated, so
      # attempting to iterate over it for old singular filed protos will yield
      # a type error.
      instance.nodes[proto.exit_block]['exit'] = True
    # Create the edges from protos.
    for edge in proto.edge:
      instance.add_edge(edge.src_index, edge.dst_index)
    # Validate the proto.
    return instance.ValidateControlFlowGraph(strict=False)

  def __eq__(self, other) -> bool:
    """Compare control flow graphs.

    This performs a "logical" comparison between graphs, excluding attributes
    that do not affect the graph, e.g. the name of the graph.

    Args:
      other: Another control flow graph.

    Returns:
      True if graphs are equal, else false.
    """
    if not isinstance(other, self.__class__):
      return False

    if self.number_of_nodes() != other.number_of_nodes():
      return False
    if self.number_of_edges() != other.number_of_edges():
      return False

    if list(self.nodes) != list(other.nodes):
      return False

    # Compare node data.
    for i in self.nodes:
      # We may want to exclude the 'name' attribute from comparisons, assuming
      # it has no logical meaning.
      if self.nodes[i] != other.nodes[i]:
        return False

    if list(self.edges) != list(other.edges):
      return False

    for i, j in self.edges:
      # Compare edge data.
      if self.edges[i, j] != other.edges[i, j]:
        return False

    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self) -> int:
    """Return the numeric hash of the instance."""
    # The hash is based on the graph topology and node and edge attributes.
    return hash((tuple(self.nodes), tuple(
        self.edges), tuple([str(self.nodes[n]) for n in self.nodes]),
                 tuple([str(self.edges[i, j]) for i, j in self.edges])))

  def IsomorphicHash(self) -> int:
    """Return a numeric hash of the graph shape."""
    # The hash is based on the nodes and edges, not their attributes.
    return hash((tuple(self.nodes), tuple(self.edges)))
