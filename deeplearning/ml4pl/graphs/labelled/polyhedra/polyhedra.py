"""Module for labelling program graphs with alias sets."""
import random
import typing

import networkx as nx
import numpy as np
import pydot
from labm8 import app
from labm8 import decorators

from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util
from deeplearning.ml4pl.graphs.unlabelled.cdfg import control_and_data_flow_graph as cdfg

FLAGS = app.FLAGS


def RecursePydot(subgraph: pydot.Dot, func: typing.Callable[[pydot.Dot, typing.Any], None],
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
      if node.get_name() in [str(r)[:-1] for r in region]:  # Need to cut off semicolon
        return {'polyhedral': True}
      
    return {'polyhedral': False}
  
  def OnInstruction(self, node_attrs: typing.Dict[str, typing.Any],
                    instruction: str) -> typing.Dict[str, typing.Any]:
    if 'polyhedral' in node_attrs:
      return {'polyhedral': node_attrs['polyhedral']}
    
    return {'polyhedral': False}

  def OnIdentifier(self, stmt_node: typing.Dict[str, typing.Any],
                   identifier_node: typing.Dict[str, typing.Any],
                   definition_type: str) -> typing.Dict[str, typing.Any]:
    if definition_type == 'def':
      if 'polyhedral' in stmt_node:
        return {'polyhedral': stmt_node['polyhedral']}

    # TODO(talbn): Perhaps no need for definition_type == 'use' (may come from outside region)
    
    return {}
  

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
      if 'polyhedral' not in ndata or ndata['polyhedral'] is False:
        continue

      # Find matching node for ndata in the original graph
      if 'original_text' not in ndata:
        gnode = node
      else:
        # This overcomes an issue with mismatching unique node identifiers between
        # an LLVM CFG and a SCoP graph
        try:
          gnode = next(n for n, d in g.nodes(data=True) if 'original_text' in d
                       and d['original_text'] == ndata['original_text'])
        except StopIteration:
          gnode = node
          
      if node not in g.nodes:
        raise ValueError(f"Entity `{node}` not found in graph, {g.nodes(data=True)}")
      g.nodes[node][y_label] = true

      
def MakePolyhedralGraphs(
    g: nx.MultiDiGraph,
    bytecode: str,
    n: typing.Optional[int] = None,
    false=False,
    true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Create up to `n` annotated graphs from a bytecode that potentially contains 
     polyhedral loops.

  Args:
    g: The unlabelled input graph.
    bytecode: The bytecode which produced the input graph.
    n: The maximum number of graphs to produce. Multiple graphs are produced by
      selecting different root pointers for alias sets. If `n` is provided,
      the number of graphs generated will be in the range
      1 <= x <= min(num_polyhedral_regions, n), where the first term is the number of
      detected polyhedral regions. If n is None, num_polyhedral_regions
      graphs will be produced.
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
  
  # One-hot encoding
  false = np.array([1, 0], np.int32)
  true = np.array([0, 1], np.int32)
  
  # Build the polyhedral building blocks
  scop_graphs, _ = opt_util.DotGraphsFromBytecode(
    bytecode, ['-O3', '-polly-process-unprofitable', '-polly-optimized-scops', '-polly-dot'])
  
  
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
                 if 'polyhedral' in node and node['polyhedral'] is True)
    if n is None:
      max_steps = max(max_steps, steps)
      cdfgs.append(annotated_cdfg)
    else:
      max_steps = steps
      cdfgs = [annotated_cdfg]
      
    # TODO(talbn): Not sure if looping is a good idea. We are potentially annotating
    #              polyhedral parts of the graph as non-polyhedral.
    #              The alternative is to yield one graph.
    if n is not None:
      labelled = g.copy()
      labelled.data_flow_max_steps_required = max_steps
      AnnotatePolyhedra(labelled, cdfgs, false=false, true=true)
      yield labelled

      if i >= n:
        break

  if n is None:
    labelled = g.copy()
    labelled.data_flow_max_steps_required = max_steps
    AnnotatePolyhedra(labelled, cdfgs, false=false, true=true)
    yield labelled
