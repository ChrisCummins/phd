"""Module for labelling program graphs with dominator trees information."""
import typing

import networkx as nx
from labm8 import app
from labm8 import decorators

from deeplearning.ml4pl.graphs import graph_query as query

FLAGS = app.FLAGS


@decorators.timeout(seconds=60)
def AnnotateDominatorTree(g: nx.MultiDiGraph,
                          root_node: str,
                          x_label: str = 'x',
                          y_label: str = 'y',
                          true: typing.Any = True,
                          false: typing.Any = False) -> typing.Tuple[int, int]:
  # Create a map from nodes to predecessors.
  predecessors: typing.Dict[str, typing.Set[str]] = {
      n: set([
          p for p in query.StatementNeighbors(
              g, n, direction=lambda src, dst: src)
      ]) for n in g.nodes()
  }

  # Initialize the dominator sets.
  dominators: typing.Dict[str, typing.Set[int]] = {
      n: set(predecessors.keys()) - set([root_node]) for n in g.nodes()
  }
  dominators[root_node] = set([root_node])

  data_flow_steps = 0
  changed = True
  while changed:
    data_flow_steps += 1
    changed = False
    for node in dominators.keys() - set([root_node]):
      dom_pred = [dominators[p] for p in predecessors[node]]
      if dom_pred:
        dom_pred = set.intersection(*dom_pred)
      else:
        dom_pred = set()
      new_dom = set([node]).union(dom_pred)
      if new_dom != dominators[node]:
        dominators[node] = new_dom
        changed = True

  num_dominated = 0
  for node, data in g.nodes(data=True):
    data[x_label] = 0
    if root_node in dominators[node]:
      num_dominated += 1
      data[y_label] = true
    else:
      data[y_label] = false

  g.nodes[root_node][x_label] = 1

  return num_dominated, data_flow_steps


def MakeDominatorTreeGraphs(
    g: nx.MultiDiGraph,
    n: typing.Optional[int] = None,
    false=False,
    true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` dominator trees from the given unlabelled graph.

  Args:
    g: The unlabelled input graph.
    n: The maximum number of graphs to produce. Multiple graphs are produced by
      selecting different root nodes for creating labels. If `n` is provided,
      the number of graphs generated will be in the range
      1 <= x <= min(num_statements, n). Else, the number of graphs will be equal
      to num_statements (i.e. one graph for each statement in the input graph).
    false: The value to set for binary false values on X and Y labels.
    true: The value to set for binary true values on X and Y labels.

  Returns:
    A generator of annotated graphs, where each graph has 'x' and 'y' labels on
    the statement nodes, and additionally 'dominated_node_count' and
    'data_flow_max_steps_required' attributes.
  """
  root_statements = query.SelectRandomNStatements(g, n)

  for root_node in root_statements[:n]:
    domtree = g.copy()
    domtree.dominated_node_count, domtree.data_flow_max_steps_required = (
        AnnotateDominatorTree(domtree, root_node, false=false, true=true))
    yield domtree
