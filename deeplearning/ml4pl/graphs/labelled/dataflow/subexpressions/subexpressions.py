"""Module for labelling program graphs with common subexpressions."""
import collections
import random
import typing

import networkx as nx

from labm8.py import app
from labm8.py import decorators

FLAGS = app.FLAGS

app.DEFINE_integer(
  "expression_set_min_size",
  2,
  "The minimum number of common subexpressions in a set to be used as a "
  "labelled example.",
)


@decorators.timeout(seconds=60)
def GetExpressionSets(g: nx.MultiDiGraph, false: typing.Any = False):
  """An expression is a statement and a list of operands, in order.

  Traverse the full graph to and build a mapping from expressions to a list of
  nodes which evaluate this expression.
  """
  # A map from expression to a list of nodes which evaluate this expression.
  expression_sets = collections.defaultdict(list)

  for node, data in g.nodes(data=True):
    data["x"] = [data["x"], 0]  # Mark expression as not the root node.
    data["y"] = false  # Mark expression as not part of a set.
    if data["type"] != "statement":
      continue

    # Build a tuple of operand position and operand values.
    position_operand_pairs = []
    for src, _, edge in g.in_edges(node, data=True):
      if edge["flow"] == "data":
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
    statement = data["text"]
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
    expression_sets[expression].append(node)

  return {k: list(sorted(set(v))) for k, v in expression_sets.items()}


def AnnotateCommonSubexpressions(
  g: nx.MultiDiGraph, root_expression: str, expression_sets, true=True
):
  """Add annotations for the given root expression. This assumes that the node
  x and y labels have been set to false, as in GetExpressionSets()
  """
  # Mark the root expression.
  root_node = random.choice(expression_sets[root_expression])
  g.nodes[root_node]["x"] = [g.nodes[root_node]["x"][0], 1]

  for node in expression_sets[root_expression]:
    g.nodes[node]["y"] = true


def MakeSubexpressionsGraphs(
  g: nx.MultiDiGraph, n: typing.Optional[int] = None, false=False, true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` subexpressions graphs from the given unlabelled graph.

  Args:
    g: The unlabelled input graph.
    n: The maximum number of graphs to produce. Multiple graphs are produced by
      selecting different root nodes for creating labels. If `n` is provided,
      the number of graphs generated will be in the range
      1 <= x <= min(num_statements, n). Else, the number of graphs is determined
      by the number of expression sets with more than --expression_set_min_size
      members.
    false: The value to set for binary false values on X and Y labels.
    true: The value to set for binary true values on X and Y labels.

  Returns:
    A generator of annotated graphs, where each graph has 'x' and 'y' labels on
    the statement nodes, and additionally 'live_out_count' and
    'data_flow_max_steps_required' attributes.
  """
  expression_sets = GetExpressionSets(g, false=false)

  # Filter expression sets based on minimum size.
  expression_sets = {
    expression: nodes
    for expression, nodes in expression_sets.items()
    if len(nodes) >= FLAGS.expression_set_min_size
  }

  expressions = list(expression_sets.keys())
  n = n or len(expressions)
  if len(expressions) > n:
    random.shuffle(expressions)
    expressions = expressions[:n]

  for root_expression in expressions:
    labelled = g.copy()
    AnnotateCommonSubexpressions(
      labelled, root_expression, expression_sets, true=true
    )
    yield labelled
