"""Module for labelling program graphs with data depedencies."""
import collections
import typing

import networkx as nx

from deeplearning.ml4pl.graphs import graph_iterators as iterators
from deeplearning.ml4pl.graphs import graph_query as query
from labm8.py import app
from labm8.py import decorators

FLAGS = app.FLAGS

app.DEFINE_boolean(
  "only_entry_blocks_for_liveness_root_nodes",
  False,
  "Use only CFG entry blocks as the root nodes for creating liveness graphs.",
)


def GetDefsAndSuccessors(g: nx.MultiDiGraph, node: str) -> typing.List[str]:
  defs, successors = [], []
  for src, dst, flow in g.out_edges(node, data="flow", default="control"):
    if flow == "data":
      defs.append(dst)
    elif flow == "control":
      successors.append(dst)
  return defs, successors


def GetUsesAndPredecessors(g: nx.MultiDiGraph, node: str) -> typing.List[str]:
  uses, predecessors = [], []
  for src, dst, flow in g.in_edges(node, data="flow", default="control"):
    if flow == "data":
      uses.append(src)
    elif flow == "control":
      predecessors.append(src)
  return uses, predecessors


@decorators.timeout(seconds=60)
def AnnotateLiveness(
  g: nx.MultiDiGraph,
  root_node: str,
  x_label: str = "x",
  y_label: str = "y",
  true: typing.Any = True,
  false: typing.Any = False,
) -> typing.Tuple[int, int]:
  # Liveness analysis begins at the exit block and works backwards. Since we
  # can't guarantee that input graphs have a single exit point, add a temporary
  # exit block which we will remove after computing liveness results.
  exit_blocks = list(node for node, _ in iterators.ExitBlockIterator(g))
  if not exit_blocks:
    raise ValueError("No exit blocks")
  g.add_node("__liveness_starting_point__", type="statement")
  for node in exit_blocks:
    g.add_edge(node, "__liveness_starting_point__", flow="control")

  # Ignore the __liveness_start_point__ block when totalling up the data flow
  # steps.
  data_flow_steps = -1

  in_sets = {node: set() for node in g.nodes()}
  out_sets = {node: set() for node in g.nodes()}

  work_list = collections.deque(["__liveness_starting_point__"])
  while work_list:
    data_flow_steps += 1
    node = work_list.popleft()
    defs, successors = GetDefsAndSuccessors(g, node)
    uses, predecessors = GetUsesAndPredecessors(g, node)

    # LiveOut(n) = U {LiveIn(p) for p in succ(n)}
    new_out_set = set().union(*[in_sets[p] for p in successors])

    # LiveIn(n) = Gen(n) U {LiveOut(n) - Kill(n)}
    new_in_set = set(uses).union(new_out_set - set(defs))

    # No need to visit predecessors if the in-set is non-empty and has not
    # changed.
    #
    # Is it possible to exit early when root_node is not the entry block?
    if not new_in_set or new_in_set != in_sets[node]:
      work_list.extend([p for p in predecessors if p not in work_list])

    # Per-step state printout for debugging.
    # print(
    #     'Step: %d:\n  processing: %s\n  uses: %s,  defs: %s\n  '
    #     'out-state %s,  new out-state %s\n  '
    #     'old in-state: %s,  new in-state: %s\n  '
    #     'predecessors: %s,  work list: %s' %
    #     (data_flow_steps, node, uses, defs, out_sets[node], new_out_set,
    #      in_sets[node], new_in_set, predecessors, work_list))

    in_sets[node] = new_in_set
    out_sets[node] = new_out_set

  # Remove the temporary singular exit block.
  g.remove_node("__liveness_starting_point__")

  # Now that we've computed the liveness results, annotate the graph.
  for _, data in g.nodes(data=True):
    data[x_label] = [data[x_label], 0]
    data[y_label] = false
  g.nodes[root_node][x_label] = [g.nodes[root_node][x_label][0], 1]

  for node in out_sets[root_node]:
    g.nodes[node][y_label] = true

  return len(out_sets[root_node]), data_flow_steps


def MakeLivenessGraphs(
  g: nx.MultiDiGraph, n: typing.Optional[int] = None, false=False, true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` liveness graphs from the given unlabelled graph.

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
    the statement nodes, and additionally 'live_out_count' and
    'data_flow_max_steps_required' attributes.
  """
  if FLAGS.only_entry_blocks_for_liveness_root_nodes:
    root_statements = [node for node, _ in iterators.EntryBlockIterator(g)]
  else:
    root_statements = query.SelectRandomNStatements(g, n)

  for root_node in root_statements[:n]:
    labelled = g.copy()
    (
      labelled.live_out_count,
      labelled.data_flow_max_steps_required,
    ) = AnnotateLiveness(labelled, root_node, false=false, true=true)
    yield labelled
