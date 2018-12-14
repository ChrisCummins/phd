"""A class representing a control flow graph."""
import typing

import networkx as nx
from absl import flags


FLAGS = flags.FLAGS


class ControlFlowGraph(nx.DiGraph):
  """A control flow graph.

  For a control flow graph to be considered "valid", the following properties
  should be adhered to:

   * All nodes in the graph should have a unique "name" attribute. This can be
     set at creation time, e.g.: cfg.add_node(0, name='foo').
   * The graph should be fully connected.

  Use the IsValidControlFlowGraph() method to check if a graph instance has
  these properties.
  """

  def __init__(self, name: str = "cfg"):
    super(ControlFlowGraph, self).__init__(name=name)

  def IsReachable(self, src, dst) -> bool:
    """Return whether dst node is reachable from src."""
    # TODO(cec): It seems that descendants() does not include self loops, so
    # test for the node in both descendants and self loops.
    return ((dst in nx.descendants(self, src)) or
            (dst in self.nodes_with_selfloops()))

  def Reachables(self, src) -> typing.Iterator[bool]:
    """Return whether each node is reachable from the src node."""
    return (self.IsReachable(src, dst) for dst in self.nodes)

  def IsValidControlFlowGraph(self) -> bool:
    """Return true if the graph is a valid control flow graph."""
    number_of_nodes = self.number_of_nodes()
    # CFGs must contain a node.
    if not number_of_nodes:
      return False
    # CFGs must be fully connected.
    # TODO:
    # if nx.number_connected_components(self) != number_of_nodes:
    #   return False
    # All nodes must have a name.
    if not all('name' in self.nodes[node] for node in self.nodes):
      return False
    # All node names must be unique.
    node_names_set = set(self.nodes[n]['name'] for n in self.nodes)
    if len(node_names_set) != number_of_nodes:
      return False
    return True
