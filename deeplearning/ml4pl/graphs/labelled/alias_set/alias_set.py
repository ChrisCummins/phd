"""Module for labelling program graphs with alias sets."""
import random
import typing

import networkx as nx
from labm8 import app
from labm8 import decorators

from compilers.llvm import opt_util

FLAGS = app.FLAGS

app.DEFINE_integer(
    'max_instances_per_alias_set', 3,
    'The maximum number of instances to produce for a single '
    'alias set. Multiple instances are produced by changing '
    'the root identifier.')


@decorators.timeout(seconds=120)
def AnnotateAliasSet(g: nx.MultiDiGraph,
                     alias_set: opt_util.AliasSet,
                     root_identifier: str,
                     x_label: str = 'x',
                     y_label: str = 'y',
                     true: typing.Any = True,
                     false: typing.Any = False) -> int:
  if root_identifier not in alias_set:
    raise ValueError(f"Root identifier {root_identifier} not in alias set")
  if root_identifier not in g.nodes:
    raise ValueError(f"Root identifier {root_identifier} not in graph")

  for _, data in g.nodes():
    data[x_label] = false
    data[y_label] = false

  g.nodes[root_identifier][x_label] = true

  for pointer in alias_set.pointers:
    if pointer.identifier not in g.nodes:
      raise ValueError(f"Identifier `{pointer.identifier}` from alias set not "
                       "found in graph")
    g.nodes[pointer.identifier][y_label] = true

  return len(alias_set.pointers)


def MakeAliasSetGraphs(
    g: nx.MultiDiGraph,
    alias_sets: typing.List[opt_util.AliasSet],
    n: typing.Optional[int] = None,
    false=False,
    true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` dependency graphs from the given unlabelled graph.

  Args:
    g: The unlabelled input graph.
    alias_sets: A list of alias sets for this graph.
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
  alias_sets = [s for s in alias_sets if len(s.pointers) > 1]
  if len(alias_sets) > n:
    random.shuffle(alias_sets)

  for alias_set in alias_sets[:n]:
    root_pointers = alias_set.pointers
    if len(root_pointers) > FLAGS.max_instances_per_alias_set:
      random.shuffle(root_pointers)

    for root_pointer in root_pointers[:FLAGS.max_instances_per_alias_set]:
      labelled = g.copy()
      labelled.alias_set_size = (AnnotateAliasSet(labelled,
                                                  alias_set,
                                                  root_pointer,
                                                  false=false,
                                                  true=true))
      yield labelled
