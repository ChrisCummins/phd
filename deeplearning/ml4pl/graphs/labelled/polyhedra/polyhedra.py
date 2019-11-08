"""Module for labelling program graphs with polyhedral SCoPs."""
import random
import typing

import networkx as nx
import numpy as np
import pydot
from labm8 import app
from labm8 import decorators

from compilers.llvm import opt
from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs.unlabelled.cdfg import control_and_data_flow_graph as cdfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util

FLAGS = app.FLAGS


def RecursePydot(subgraph: pydot.Dot,
                 func: typing.Callable[[pydot.Dot, typing.Any], None],
                 state: typing.Any):
  func(subgraph, state)
  for ss in subgraph.get_subgraphs():
    RecursePydot(ss, func, state)


def SubNodes(subgraph: pydot.Dot, nodes: typing.List[typing.Any]):
  nodes.extend(subgraph.get_nodes())


def GetSubgraph(subgraph: pydot.Dot, state: typing.Dict[pydot.Dot, typing.Any]):
  if subgraph.get('style') == 'filled':
    nodes = []
    RecursePydot(subgraph, SubNodes, nodes)
    state[subgraph] = nodes


class PolyhedralRegionAnnotator(llvm_util.TagHook):
  """Tag hook that annotates polyhedral regions on the nodes (with the attribute 
  `polyhedral=True`)"""

  def OnGraphBegin(self, dot: pydot.Dot):
    # Get polyhedral basic blocks from Polly and pydot
    # Obtain all basic blocks in polyhedral region (need to recurse into sub-subgraphs)
    self.regions = {}
    RecursePydot(dot, GetSubgraph, self.regions)

  def OnNode(self, node: pydot.Node) -> typing.Dict[str, typing.Any]:
    for region in self.regions.values():
      if node.get_name() in [str(r)[:-1] for r in region
                            ]:  # Need to cut off semicolon
        return {'polyhedral': True}

    return {'polyhedral': False}

  def OnInstruction(self, node_attrs: typing.Dict[str, typing.Any],
                    instruction: str) -> typing.Dict[str, typing.Any]:
    return {'polyhedral': node_attrs.get('polyhedral', False)}

  def OnIdentifier(self, stmt_node: typing.Dict[str, typing.Any],
                   identifier_node: typing.Dict[str, typing.Any],
                   definition_type: str) -> typing.Dict[str, typing.Any]:
    if definition_type == 'def':
      if 'polyhedral' in stmt_node:
        return {'polyhedral': stmt_node['polyhedral']}

    # TODO(talbn): Perhaps no need for definition_type == 'use' (may come from outside region)

    return {}


def BytecodeToPollyCanonicalized(source: str) -> str:
  process = opt.Exec(['-polly-canonicalize', '-S', '-', '-o', '-'],
                     stdin=source)
  if process.returncode:
    raise opt.OptException(
        'Error in canonicalization opt execution (%d)' % process.returncode)
  return process.stdout


def CreateCDFG(bytecode: str) -> nx.MultiDiGraph:
  builder = cdfg.ControlAndDataFlowGraphBuilder()
  return builder.Build(bytecode)


@decorators.timeout(seconds=120)
def AnnotatePolyhedra(g: nx.MultiDiGraph,
                      annotated_cdfgs: typing.List[nx.MultiDiGraph],
                      x_label: str = 'x',
                      y_label: str = 'y',
                      false=False,
                      true=True) -> None:
  """

  Args:
    g: The graph.
    annotated_cdfgs: CDFGs with nodes that polly marked as "polyhedral" (green in input dot).
    x_label: The graph 'x' attribute property attribute name.
    y_label: The graph 'y' attribute property attribute name.
    false: The value to set for nodes that are not polyhedral.
    true: The value to set for nodes that are polyhedral.
  """

  # Set all of the nodes as not-polyhedral at first.
  # X labels are a list which concatenates the original graph 'x'
  # embedding indices with a [0,1] value for false/true, respectively.
  for _, data in g.nodes(data=True):
    data[x_label] = [data[x_label], 0]
    data[y_label] = false

  # Obtain nodes in g (slimmed down version of CDFG generation, only nodes)
  for cdfg in annotated_cdfgs:
    # Mark the nodes in the polyhedral regions
    for node, ndata in cdfg.nodes(data=True):
      if not ndata.get('polyhedral'):
        continue

      if node not in g.nodes:
        raise ValueError(
            f"Entity `{node}` not found in graph, {g.nodes(data=True)}")
      g.nodes[node][y_label] = true


def MakePolyhedralGraphs(
    bytecode: str,
    n: typing.Optional[int] = None,
    false=False,
    true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Create an annotated graph from a bytecode that potentially contains 
     polyhedral loops.

  Args:
    g: The unlabelled input graph.
    bytecode: The bytecode which produced the input graph.
    n: The maximum number of graphs to produce. This value is ignored and one graph 
       will be produced with all polyhedral regions annotated.
    false: TODO(cec): Unused. This method is hardcoded to use 2-class 1-hots.
    true: TODO(cec): Unused. This method is hardcoded to use 2-class 1-hots.

  Returns:
    A generator of annotated graphs, where each graph has 'x' and 'y' labels on
    the statement nodes, and additionally a 'data_flow_max_steps_required'
    attribute which is set to the largest number of statements in a polyhedral block.
  """
  # TODO(cec): Replace true/false args with a list of class values for all
  # graph annotator functions.
  del false
  del true
  del n

  # One-hot encoding
  false = np.array([1, 0], np.int32)
  true = np.array([0, 1], np.int32)

  # Canonicalize input graph (see http://polly.llvm.org/docs/Architecture.html)
  bytecode = BytecodeToPollyCanonicalized(bytecode)
  g = CreateCDFG(bytecode)

  # Build the polyhedral building blocks
  scop_graphs, _ = opt_util.DotGraphsFromBytecode(bytecode, [
      '-O1', '-polly-process-unprofitable', '-polly-optimized-scops',
      '-polly-dot', '-polly-optimizer=none'
  ])

  # Loop over each function
  max_steps = 0
  cdfgs = []
  for i, graph in enumerate(scop_graphs):
    graph_annotator = PolyhedralRegionAnnotator()
    dot = graph
    cfg = llvm_util.ControlFlowGraphFromDotSource(dot, tag_hook=graph_annotator)
    builder = cdfg.ControlAndDataFlowGraphBuilder()
    annotated_cdfg = builder.BuildFromControlFlowGraph(cfg)

    steps = sum(1 for nid, node in annotated_cdfg.nodes(data=True)
                if node.get('polyhedral'))
    max_steps = max(max_steps, steps)
    cdfgs.append(annotated_cdfg)

  labelled = g.copy()
  labelled.data_flow_max_steps_required = max_steps
  AnnotatePolyhedra(labelled, cdfgs, false=false, true=true)
  yield labelled
