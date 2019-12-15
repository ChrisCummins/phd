"""Module for labelling program graphs with common subexpressions."""
import collections
from typing import List
from typing import Set

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_integer(
  "expression_set_min_size",
  2,
  "The minimum number of common subexpressions in a set to be used as a "
  "labelled example.",
)

NOT_COMMON_SUBEXPRESSION = [1, 0]
COMMON_SUBEXPRESSION = [0, 1]


def GetExpressionSets(g: nx.MultiDiGraph) -> List[Set[int]]:
  """An expression is a statement and a list of operands, in order.

  Traverse the full graph to and build a mapping from expressions to a list of
  nodes which evaluate this expression.

  Returns:
    A list of expression sets, where each expression set is a collection of
    nodes that each evaluate the same expression.
  """
  # A map from expression to a list of nodes which evaluate this expression.
  expression_sets = collections.defaultdict(set)

  for node, data in g.nodes(data=True):
    if data["type"] != programl_pb2.Node.STATEMENT:
      continue

    # Build a tuple of operand position and operand values.
    position_operand_pairs = []
    for src, _, edge in g.in_edges(node, data=True):
      if edge["flow"] == programl_pb2.Edge.DATA:
        position_operand_pairs.append((edge["position"], src))

    # A statement without operands can never be a common subexpression.
    if not position_operand_pairs:
      continue

    # If the operands are commutative, sort operands by their name. For,
    # non-commutative operands, sort the operand position (i.e. order).
    # E.g.
    #    '%3 = add %2 %1' == '%4 = add %1 %2'  # commutative
    # but:
    #    '%3 = sub %2 %1' != '%4 = sub %1 %2'  # non-commutative
    #
    # Commutative statements derived from:
    # <https://llvm.org/docs/LangRef.html#instruction-reference>.
    statement = data["preprocessed_text"]
    # Strip the lhs from the instruction, if any.
    if " = " in statement:
      statement = statement[statement.index(" = ") + 3 :]
    if (  # Binary operators:
      statement.startswith("add ")
      or statement.startswith("fadd ")
      or statement.startswith("mul ")
      or statement.startswith("fmul ")
      or
      # Bitwise binary operators:
      statement.startswith("and ")
      or statement.startswith("xor ")
      or statement.startswith("or ")
    ):
      # Commutative statement, order by identifier name.
      position_operand_pairs = sorted(
        position_operand_pairs, key=lambda x: x[1]
      )
    else:
      # Non-commutative statement, order by position.
      position_operand_pairs = sorted(
        position_operand_pairs, key=lambda x: x[0]
      )

    operands = tuple(x[1] for x in position_operand_pairs)

    # An expression is a statement and a list of ordered operands.
    expression = (statement, operands)

    # Add the statement node to the expression lists table.
    expression_sets[expression].add(node)

  return list(expression_sets.values())


class CommonSubexpressionAnnotator(
  data_flow_graphs.NetworkXDataFlowGraphAnnotator
):
  """Annotate graphs with common subexpression information."""

  def __init__(self, *args, **kwargs):
    self._expression_sets = None
    self._root_nodes = None
    super(CommonSubexpressionAnnotator, self).__init__(*args, **kwargs)

  def IsValidRootNode(self, node: int, data) -> bool:
    """Determine if the given node should be used as a root node.

    For subexpressions, a valid root node is one which is a part of an
    expression set of at least --expression_set_min_size expressions.
    """
    del data
    # Lazily evaluate the set of root nodes.
    if self._root_nodes is None:
      self._root_nodes = set()
      for expression_set in self.expression_sets:
        if len(expression_set) >= FLAGS.expression_set_min_size:
          self._root_nodes = self._root_nodes.union(expression_set)
    return node in self._root_nodes

  @property
  def expression_sets(self):
    """Lazily compute expression sets."""
    if self._expression_sets is None:
      self._expression_sets = GetExpressionSets(self.g)
    return self._expression_sets

  def Annotate(self, g: nx.MultiDiGraph, root_node: int) -> None:
    g.graph["data_flow_root_node"] = root_node

    for expression_set in self.expression_sets:
      if root_node in expression_set:
        break
    else:
      g.graph["data_flow_steps"] = 0
      g.graph["data_flow_positive_node_count"] = 0
      return

    for _, data in g.nodes(data=True):
      data["x"].append(data_flow_graphs.ROOT_NODE_NO)
      data["y"] = NOT_COMMON_SUBEXPRESSION

    for node in expression_set:
      g.nodes[node]["y"] = COMMON_SUBEXPRESSION

    g.nodes[root_node]["x"][-1] = data_flow_graphs.ROOT_NODE_YES

    g.graph["data_flow_steps"] = 2
    g.graph["data_flow_positive_node_count"] = len(expression_set)
