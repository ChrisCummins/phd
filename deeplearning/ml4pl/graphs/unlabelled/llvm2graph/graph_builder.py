"""This module defines the ProGraML graph representation.

A ProGraML graph is a directed multigraph which is the union a control flow,
data flow, and call graphs.
"""
import copy
import difflib
import itertools
import pickle
import re
import typing

import networkx as nx
import numpy as np

from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs import graph_iterators as iterators
from deeplearning.ml4pl.graphs import graph_query as query
from deeplearning.ml4pl.graphs.unlabelled import programl_pb2
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util
from deeplearning.ml4pl.graphs.unlabelled.cg import call_graph as cg
from deeplearning.ncc import rgx_utils as rgx
from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8 import app
from labm8 import bazelutil
from labm8 import decorators

FLAGS = app.FLAGS


def MakeUndefinedFunctionGraph(function_name: str,
                               dictionary) -> nx.MultiDiGraph:
  """Create an empty function with the given name.

  Use this to create function graphs for undefined functions. The generated
  functions consist only of an entry and exit block, with a control edge
  between them.
  """
  g = nx.MultiDiGraph()

  g.name = function_name
  g.entry_block = f'{function_name}_entry'
  g.exit_block = f'{function_name}_exit'

  g.add_node(g.entry_block,
             type='statement',
             function=function_name,
             text='!UNK',
             original_text=g.entry_block,
             x=dictionary['!UNK'])
  g.add_node(g.exit_block,
             type='statement',
             function=function_name,
             text='!UNK',
             original_text=g.exit_block,
             x=dictionary['!UNK'])
  g.add_edge(g.entry_block,
             g.exit_block,
             function=function_name,
             flow='control',
             position=0)

  return g


def InsertFunctionGraph(graph, function_name, function_graphs,
                        dictionary) -> typing.Tuple[nx.MultiDiGraph, str, str]:
  """Insert the named function graph to the graph."""
  if function_name not in function_graphs:
    function_graphs[function_name] = MakeUndefinedFunctionGraph(
        function_name, dictionary)

  function_graph = copy.deepcopy(function_graphs[function_name])
  graph = nx.compose(graph, function_graph)
  return graph, function_graph.entry_block, function_graph.exit_block


def AddInterproceduralCallEdges(
    graph: nx.MultiDiGraph, call_multigraph: nx.MultiDiGraph,
    function_entry_exit_nodes: typing.Dict[str, typing.Tuple[str, str]],
    get_call_site_successor: typing.Callable[[nx.MultiDiGraph, str], str]
) -> None:
  """Add "call" edges between procedures to match the call graph.

  Args:
    graph: The disconnected per-function graphs.
    call_multigraph: A call graph with parallel edges to indicate multiple calls
      from the same function.
    function_entry_exit_nodes: A mapping from function name to a tuple of entry
      and exit statements.
    get_call_site_successor: A callback which takes as input a graph and a call
      site statement within the graph, and returns the destination node for
      calls from this site.
  """
  # Drop the parallel edges by converting the call graph back to a regular
  # directed graph. Iterating over the edges in this graph then provides the
  # functions that need connecting, while the multigraph tells us how many
  # connections to expect.
  call_graph = nx.DiGraph(call_multigraph)

  for src, dst in call_graph.edges:
    # Ignore connections to the root node, we have already processed them.
    if src == 'external node':
      continue

    call_sites = query.FindCallSites(graph, src, dst)

    if not call_sites:
      continue

    # Check that the number of call sounds we found matches the expected number
    # from the call graph.
    # multigraph_call_count = call_multigraph.number_of_edges(src, dst)
    # if len(call_sites) != multigraph_call_count:
    #   raise ValueError("Call graph contains "
    #                    f"{humanize.Plural(multigraph_call_count, 'call')} from "
    #                    f"function `{src}` to `{dst}`, but found "
    #                    f"{len(call_sites)} call sites in the graph")

    for call_site in call_sites:
      if dst not in function_entry_exit_nodes:
        continue
      # Lookup the nodes to connect.
      call_entry, call_exit = function_entry_exit_nodes[dst]
      call_site_successor = get_call_site_successor(graph, call_site)
      # Connect the nodes.
      graph.add_edge(call_site, call_entry, flow='call', position=0)
      graph.add_edge(call_exit, call_site_successor, flow='call', position=0)


class ProGraMLGraphBuilder(object):

  def __init__(self,
               dataflow: bool = True,
               preprocess_text: bool = True,
               discard_unknown_statements: bool = False):
    """Instantiate a Control and Data Flow Graph (CDFG) builder.

    Args:
      dataflow: Determine the type of data flow information that is added to
        control flow graphs. If False, only control flow is used. If True,
        nodes representing identifiers and immediates are inserted, and data
        flow edges flowing between the identifier nodes and statements.
      preprocess_text: If true, pre-process the text of statements to discard
        literals, normalise identifiers, etc.
      discard_unknown_statements: Pre-processing can choose to delete statements
        (see inst2vec_preprocess.keep()). In that case, the node can either be
        removed, or the node can be kept but with the text set to `!UNK`. If the
        node is removed, the control flow paths flowing through the node are
        preserved.
    """
    self.dataflow = dataflow
    self.preprocess_text = preprocess_text
    self.discard_unknown_statements = discard_unknown_statements

  def MaybePreprocessStatementText(self, g: nx.Graph) -> None:
    """Replace 'text' statement attributes with inst2vec preprocessed.

    If preprocess_text is True, an 'original_text' attribute is added to nodes
    which stores the unmodified text.

    Args:
      g: The graph to pre-process the statement node texts of.
    """
    if not self.preprocess_text:
      return

    for node, data in iterators.StatementNodeIterator(g):
      if 'text' not in data:
        raise ValueError(
            f"No `text` attribute for node `{node}` with attributes: {data}")
    lines = [[data['text']] for _, data in iterators.StatementNodeIterator(g)]
    preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
    preprocessed_texts = [
        inst2vec_preprocess.PreprocessStatement(x[0] if len(x) else '')
        for x in preprocessed_lines
    ]
    # Pre-processing may through away lines (e.g. 'target datalayout' lines).
    # Keep track of those that have been discarded, to be removed later.
    nodes_to_remove: typing.Set[str] = set()
    edges_to_remove: typing.Set[typing.Tuple[str, str]] = set()
    edges_to_add: typing.Set[
        typing.Tuple[str, str, typing.Dict[str, typing.Any]]] = set()
    for (node, data), text in zip(g.nodes(data=True), preprocessed_texts):
      if text:
        data['original_text'] = data['text']
        data['text'] = text
        data['type'] = 'statement'
        data['x'] = self.DictionaryLookup(data['text'])
      elif self.discard_unknown_statements:
        nodes_to_remove.add(node)
      else:
        data['original_text'] = data['text']
        data['text'] = '!UNK'
        data['type'] = 'statement'
        data['x'] = self.DictionaryLookup(data['text'])

    # Delete the nodes that have been discarded by preprocessing, and re-connect
    # any edges that flow through the nodes.
    for node in nodes_to_remove:
      in_edges = g.in_edges(node)
      out_edges = g.out_edges(node)
      in_nodes = iterators.SuccessorNodes(g,
                                          node,
                                          ignored_nodes=nodes_to_remove,
                                          direction=lambda src, dst: src)
      out_nodes = iterators.SuccessorNodes(g,
                                           node,
                                           ignored_nodes=nodes_to_remove,
                                           direction=lambda src, dst: dst)

      for edge in in_edges:
        edges_to_remove.add(edge)
      for edge in out_edges:
        edges_to_remove.add(edge)
      for src, dst in itertools.product(in_nodes, out_nodes):
        # Note this loses position information.
        edges_to_add.add((src, dst, {'position': 0}))

    for edge in edges_to_remove:
      g.remove_edge(*edge)
    for node in nodes_to_remove:
      g.remove_node(node)
    for src, dst, data in edges_to_add:
      g.add_edge(src, dst, **data)

  def MaybeAddSingleEntryBlock(self, g: nx.MultiDiGraph) -> None:
    """Add a magic entry block."""
    entry_blocks = list(iterators.EntryBlockIterator(g))
    if not entry_blocks:
      raise ValueError("No entry blocks found in graph!")

    if (self.only_add_entry_and_exit_blocks_if_required and
        len(entry_blocks) == 1):
      entry_block = entry_blocks[0][0]
    else:
      entry_block = f'{g.name}_entry'
      g.add_node(entry_block,
                 name=entry_block,
                 type='magic',
                 x=self.dictionary['!MAGIC'])
      for node, data in entry_blocks:
        g.add_edge(entry_block, node, flow='control')
    g.entry_block = entry_block

  def MaybeAddSingleExitBlock(self, g: nx.MultiDiGraph) -> None:
    """Add a magic exit block to unite the exit statements of a CFG.

    Args:
      g: The graph to add the exit block to.
    """
    exit_blocks = list(iterators.ExitBlockIterator(g))
    if not exit_blocks:
      raise ValueError("No exit blocks found in graph!")

    if (self.only_add_entry_and_exit_blocks_if_required and
        len(exit_blocks) == 1):
      exit_block = exit_blocks[0][0]
    else:
      exit_block = f'{g.name}_exit'
      g.add_node(exit_block,
                 name=exit_block,
                 type='magic',
                 x=self.dictionary['!MAGIC'])
      # Connect exit blocks.
      for node, data in exit_blocks:
        g.add_edge(node, exit_block, flow='control')
      # Add a dataflow edge out, if there is one.
      for src, dst, data in iterators.DataFlowEdgeIterator(g):
        if dst == node:
          g.add_edge(node, exit_block, flow='data')
          break
    g.exit_block = exit_block

  def MaybeAddDataFlowElements(
      self, g: nx.MultiDiGraph,
      tag_hook: typing.Optional[llvm_util.TagHook]) -> None:
    if self.dataflow == 'none':
      return

    prefix = lambda s: f'{g.name}_{s}'
    unprefix = lambda s: s[len(f'{g.name}_'):]

    # Collect the edges to add so that we don't modify the graph while
    # iterating.
    edges_to_add: typing.List[typing.Tuple[str, str, str, int]] = []

    for statement, data in iterators.StatementNodeIterator(g):
      # TODO(github.com/ChrisCummins/ml4pl/issues/6): Separate !IDENTIFIER
      # and !IMMEDIATE uses.
      def_, uses = GetLlvmStatementDefAndUses(
          data['text'], store_destination_is_def=self.store_destination_is_def)
      if def_:  # Data flow out edge.
        def_name = f'{prefix(def_)}_operand'
        edges_to_add.append(
            (statement, def_name, def_name, 0, def_, data, 'def'))
      for position, identifier in enumerate(uses):  # Data flow in edge.
        identifier_name = f'{prefix(identifier)}_operand'
        edges_to_add.append((identifier_name, statement, identifier_name,
                             position, identifier, data, 'use'))

    for src, dst, identifier, position, name, original_node, dtype in edges_to_add:
      g.add_edge(src, dst, flow='data', position=position)
      node = g.nodes[identifier]
      # TODO(github.com/ChrisCummins/ml4pl/issues/6): Separate !IDENTIFIER
      # and !IMMEDIATE nodes.
      node['type'] = 'identifier'
      node['name'] = name
      node['text'] = name
      node['x'] = self.dictionary['!IDENTIFIER']

      if tag_hook is not None:
        other_attrs = tag_hook.OnIdentifier(original_node, node, dtype) or {}
        for attrname, attrval in other_attrs.items():
          node[attrname] = attrval

  def DictionaryLookup(self, statement: str) -> int:
    if statement in self.dictionary:
      return self.dictionary[statement]
    else:
      return self.dictionary['!UNK']

  @decorators.timeout(120)
  def BuildFromControlFlowGraph(
      self, cfg: llvm_util.LlvmControlFlowGraph) -> nx.DiGraph:
    """Build a CDFG from an LLVM Control Flow Graph.

    Args:
      cfg: The control flow graph to build a CDFG from.

    Returns:
      A MultiDiGraph.
    """
    # Expand the control flow graph to a full flow graph (one block per
    # statement).
    ffg = cfg.BuildFullFlowGraph()

    # Copy the DiGraph to a MultiDiGraph, which is required for the parallel
    # control- and data-flow edges.
    #
    # While doing this, prefix the node and edge names with thename of the
    # graph so that multiple graphs from the same bytecode file can be composed.
    g = nx.MultiDiGraph()
    g.name = ffg.name
    for node, data in ffg.nodes(data=True):
      g.add_node(f'{g.name}_{node}', **data)
    for src, dst, data in ffg.edges(data=True):
      g.add_edge(f'{g.name}_{src}', f'{g.name}_{dst}', **data)

    # Record the graph entry block.
    g.entry_block = f'{g.name}_{ffg.entry_block}'

    # Set the type for existing control flow edges.
    for _, _, data in g.edges(data=True):
      data['flow'] = 'control'

    self.MaybeAddDataFlowElements(g, ffg.tag_hook)
    self.MaybePreprocessStatementText(g)
    self.MaybeAddSingleEntryBlock(g)
    self.MaybeAddSingleExitBlock(g)
    return g

  def ComposeGraphs(self, function_graphs: typing.List[nx.MultiDiGraph],
                    call_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Combine function-level graphs into a single inter-procedural graph.

    Args:
      function_graphs: The function graphs to compose. These are unmodified.
      call_graph: The call graph with which to combine the function graphs.
    """
    function_names = [g.name for g in function_graphs]
    if len(set(function_names)) != len(function_names):
      raise ValueError(
          f"Duplicate function names found: {sorted(function_names)}")

    if 'external node' not in call_graph.nodes():
      raise ValueError("Call graph missing `external node` root")

    # Add a function attribute to all nodes to track their originating function.
    for function_graph in function_graphs:
      for _, data in function_graph.nodes(data=True):
        data['function'] = function_graph.name

    function_graph_map = {g.name: g for g in function_graphs}

    # Create the inter-procedural graph with a magic root node.
    interprocedural_graph = nx.MultiDiGraph()
    interprocedural_graph.add_node('root',
                                   name='root',
                                   type='magic',
                                   x=self.dictionary['!MAGIC'])

    # Add each function to the interprocedural graph.
    function_entry_exit_nodes: typing.Dict[str, typing.Tuple[str, str]] = {}

    for _, dst in call_graph.out_edges('external node'):
      interprocedural_graph, function_entry, function_exit = InsertFunctionGraph(
          interprocedural_graph, dst, function_graph_map, self.dictionary)
      function_entry_exit_nodes[dst] = (function_entry, function_exit)

      # Connect the newly inserted function to the root node.
      interprocedural_graph.add_edge('root',
                                     function_entry,
                                     flow='call',
                                     position=0)

    if self.call_edge_returns_to_successor:
      get_call_site_successor = query.GetCallStatementSuccessor
    else:
      get_call_site_successor = lambda g, n: n

    AddInterproceduralCallEdges(interprocedural_graph, call_graph,
                                function_entry_exit_nodes,
                                get_call_site_successor)

    return interprocedural_graph

  @decorators.timeout(120)
  def Build(self,
            bytecode: str,
            opt=None,
            tag_hook: typing.Optional[llvm_util.TagHook] = None
           ) -> programl_pb2.ProgramGraph:
    """Construct an inter-procedural control- and data-flow graph.

    Args:
      bytecode: The bytecode to construct the graph from.
      opt: The path to LLVM `opt` binary to use to construct control-flow and
        call graphs from. The default uses the opt binary packaged with
        //compilers/llvm:opt.
      tag_hook: An optional object that can tag specific nodes in the graph
                according to some logic.

    Returns:
      A networkx graph.
    """
    # TODO(github.com/ChrisCummins/ProGraML/issues/2): Update and refactor from
    # old CDFG construction.

    # First construct the control flow graphs using opt.
    call_graph_dot, cfg_dots = (
        opt_util.DotCallGraphAndControlFlowGraphsFromBytecode(bytecode,
                                                              opt_path=opt))
    # Then construct the call graph dot using opt.
    call_graph = cg.CallGraphFromDotSource(call_graph_dot)
    # Construct networkx control flow graphs from the dot graphs.
    cfgs = [
        llvm_util.ControlFlowGraphFromDotSource(cfg_dot, tag_hook=tag_hook)
        for cfg_dot in cfg_dots
    ]
    # Add data flow elements to control flow graphs.
    graphs = [self.BuildFromControlFlowGraph(cfg) for cfg in cfgs]
    # Finally, compose the per-function graphs into a whole-module graph.
    return self.ComposeGraphs(graphs, call_graph)


def ToControlFlowGraph(g: nx.MultiDiGraph):
  """Create a new graph with only the statements and control flow edges."""
  # CFGs cannot have parallel edges, so we use only a DiGraph rather than
  # MultiDiGraph.
  cfg = nx.DiGraph()

  for node, data in iterators.StatementNodeIterator(g):
    cfg.add_node(node, **data)

  for src, dst, data in iterators.ControlFlowEdgeIterator(g):
    cfg.add_edge(src, dst, **data)


def SerializeToStatementList(g: nx.MultiDiGraph,
                             root: str = 'root') -> typing.Iterable[str]:
  visited_statements = set()
  visited_functions = set()

  # Maintain a list of functions to visit.
  functions = [root]

  while functions:
    function_name = functions[-1]
    functions.pop()

    visited_functions.add(function_name)

    stack = [function_name]

    # yield f'define @{stack[0]}'

    # Pre-order depth first graph traversal to emit the strings.
    while stack:
      node = stack[-1]
      stack.pop()

      if node in visited_statements:
        continue

      visited_statements.add(node)
      yield node
      for _, dst, flow in g.out_edges(node, data='flow', default='control'):
        if flow == 'control':
          if dst not in visited_statements:
            stack.append(dst)
        elif flow == 'call':
          if dst not in visited_functions:
            functions.append(dst)
