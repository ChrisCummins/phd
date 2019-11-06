"""Module for labelling program graphs with alias sets."""
import random
import typing

import networkx as nx
import numpy as np
from labm8 import app
from labm8 import decorators

from compilers.llvm import opt_util

FLAGS = app.FLAGS

app.DEFINE_integer(
    'alias_set_min_size', 2,
    'The minimum number of pointers in an alias set to be used as a labelled '
    'example.')


@decorators.timeout(seconds=120)
def AnnotateAliasSet(g: nx.MultiDiGraph,
                     root_identifier: str,
                     identifiers_in_set: typing.List[str],
                     x_label: str = 'x',
                     y_label: str = 'y',
                     false=False,
                     true=True) -> int:
  """

  Args:
    g: The graph.
    root_identifier: A name of a node in the alias set.
    identifiers_in_set: The names of the nodes in the alias set.
    x_label: The graph 'x' attribute property attribute name.
    y_label: The graph 'y' attribute property attribute name.
    false: The value to set for nodes not in the alias set.
    true: The value to set for nodes in the alias set.

  Returns:
    The number of identifiers in the alias set.
  """
  # Set all of the nodes as not the root identifier and not part of the alias
  # set. X labels are a list which concatenates the original graph 'x'
  # embedding indices with a [0,1] value for false/true, respectively.
  for _, data in g.nodes(data=True):
    data[x_label] = [data[x_label], 0]
    data[y_label] = false
  g.nodes[root_identifier][x_label] = [g.nodes[root_identifier][x_label], 1]

  # Mark the nodes in the alias set.
  for pointer in identifiers_in_set:
    if pointer not in g.nodes:
      identifier_nodes = [
          node for node, type_ in g.nodes(data='type') if type_ == 'identifier'
      ]
      raise ValueError(f"Pointer `{pointer}` not in function with identifiers "
                       f"{identifier_nodes}")
    g.nodes[pointer][y_label] = true

  return len(identifiers_in_set)


def MakeAliasSetGraphs(
    g: nx.MultiDiGraph,
    bytecode: str,
    n: typing.Optional[int] = None,
    false=False,
    true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` alias set graphs.

  Args:
    g: The unlabelled input graph.
    bytecode: The bytecode which produced the input graph.
    n: The maximum number of graphs to produce. Multiple graphs are produced by
      selecting different root pointers for alias sets. If `n` is provided,
      the number of graphs generated will be in the range
      1 <= x <= min(num_alias_sets, n), where num_alias_sets is the number of
      alias sets larger than --alias_set_min_size. If n is None, num_alias_sets
      graphs will be produced.
    false: TODO(cec): Unused. This method is hardcoded to use 3-class 1-hots.
    true: TODO(cec): Unused. This method is hardcoded to use 3-class 1-hots.

  Returns:
    A generator of annotated graphs, where each graph has 'x' and 'y' labels on
    the statement nodes, and additionally a 'data_flow_max_steps_required'
    attribute which is set to the number of pointers in the alias set.
  """
  # TODO(cec): Replace true/false args with a list of class values for all
  # graph annotator functions.
  del false
  del true

  # Build the alias sets for the given bytecode.
  alias_sets_by_function = opt_util.GetAliasSetsByFunction(bytecode)

  functions = {
      function for node, function in g.nodes(data='function')
      # Not all nodes have a 'function' attribute, e.g. the magic root node.
      if function
  }

  # Silently drop alias sets for functions which don't exist in the graph.
  deleted_alias_sets = []
  for function in alias_sets_by_function:
    if function not in functions:
      deleted_alias_sets.append(function)
      del alias_sets_by_function[function]
  if deleted_alias_sets:
    app.Warning(
        "Removed %d alias sets generated from bytecode but not found in "
        "graph: %s", len(deleted_alias_sets), deleted_alias_sets)

  function_alias_set_pairs: typing.List[
      typing.Tuple[str, opt_util.AliasSet]] = []
  # Flatten the alias set dictionary and ignore any alias sets that are smaller
  # than the threshold size.
  for function, alias_sets in alias_sets_by_function.items():
    function_alias_set_pairs += [
        (function, alias_set)
        for alias_set in alias_sets
        if len(alias_set.pointers) >= FLAGS.alias_set_min_size
    ]

  # Select `n` random alias sets to generate labelled graphs for.
  if n and len(function_alias_set_pairs) > n:
    random.shuffle(function_alias_set_pairs)
    function_alias_set_pairs = function_alias_set_pairs[:n]

  for function, alias_set in function_alias_set_pairs:
    # Translate the must/may alias property into 3-class 1-hot labels.
    if alias_set.type == 'may alias':
      false = np.array([1, 0, 0], np.int32)
      true = np.array([0, 1, 0], np.int32)
    elif alias_set.type == 'must alias':
      false = np.array([1, 0, 0], np.int32)
      true = np.array([0, 0, 1], np.int32)
    else:
      raise ValueError(f"Unknown alias set type `{alias_set.type}`")

    # Transform pointer name into the node names produced by the ComposeGraphs()
    # method in the graph builder. When we compose multiple graphs, we add the
    # function name as a prefix, and `_operand` suffix to identifier nodes.
    pointers = [
        f'{function}_{p.identifier}_operand' for p in alias_set.pointers
    ]

    root_pointer = random.choice(pointers)
    labelled = g.copy()
    labelled.data_flow_max_steps_required = AnnotateAliasSet(labelled,
                                                             root_pointer,
                                                             pointers,
                                                             false=false,
                                                             true=true)
    yield labelled
