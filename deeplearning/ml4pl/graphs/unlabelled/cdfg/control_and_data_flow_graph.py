"""This module defines the Control and Data Flow Graph (CDFG).

CDFGs are directed multigraphs in which nodes represent single LLVM
instructions, and edges show control and data flow. They are constructed from an
LLVM module.
"""
import itertools

import collections
import copy
import io
import networkx as nx
import typing

from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util
from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS


def GetAllocationStatementForIdentifier(g: nx.Graph, identifier: str) -> str:
  for node, data in StatementNodeIterator(g):
    if ' = alloca ' in data['text']:
      allocated_identifier = data['text'].split(' =')[0]
      if allocated_identifier == identifier:
        return node
  raise ValueError(
      f"Unable to find `alloca` statement for identifier `{identifier}`")


def GetLlvmStatementDestinationAndOperands(
    statement: str, store_destination_is_output: bool = True
) -> typing.Tuple[str, typing.List[str]]:
  """Get the destination identifier for an LLVM statement (if any), and a list
  of operand identifiers (if any).
  """
  destination, operands = '', []

  if '=' in statement:
    destination, statement = statement.split('=')

  m_loc, m_glob, m_label, m_label2 = inst2vec_preprocess.get_identifiers_from_line(
      statement)
  operands += m_loc + m_glob + m_label + m_label2

  # Store is a special case because it doesn't have an LHS, but writes to one
  # of its operands.
  if statement.startswith('store '):
    if len(operands) == 1:
      # store <immediate> <dst>
      store_destination = operands[0]
      operands = []
    elif len(operands) == 2:
      # store <src> <dst>
      store_destination = operands[0]
      operands = [operands[1]]
    else:
      raise ValueError(f'Unexpected number of operands ({len(operands)}) for '
                       f'store statement: `{statement}`')
    if store_destination_is_output:
      destination = store_destination

  # Strip whitespace from the strings.
  strip = lambda strings: (s.strip() for s in strings)
  return destination.strip(), strip(operands)


def NodeTypeIterator(g: nx.DiGraph, type_name: str):
  """Iterate over nodes in a graph of a given type."""
  for node, data in g.nodes(data=True):
    # Default node type is statement.
    if data.get('type', 'statement') == type_name:
      yield node, data


def StatementNodeIterator(g: nx.DiGraph):
  """Iterate over the statement nodes in a graph."""
  yield from NodeTypeIterator(g, 'statement')


def IdentifierNodeIterator(g: nx.DiGraph):
  """Iterate over identifier nodes in a graph."""
  yield from NodeTypeIterator(g, 'identifier')


def MagicNodeIterator(g: nx.DiGraph):
  """Iterate over the statement nodes in a graph."""
  yield from NodeTypeIterator(g, 'magic')


def EdgeTypeIterator(g: nx.DiGraph, type_name: str):
  for src, dst, data in g.edges(data=True):
    if data.get('flow', 'control') == type_name:
      yield src, dst, data


def ControlFlowEdgeIterator(g: nx.DiGraph):
  """Iterate over the control flow edges of a graph."""
  yield from EdgeTypeIterator(g, 'control')


def DataFlowEdgeIterator(g: nx.DiGraph):
  """Iterate over the data flow edges of a graph."""
  yield from EdgeTypeIterator(g, 'data')


def CallEdgeIterator(g: nx.DiGraph):
  """Iterate over the call edges of a graph."""
  yield from EdgeTypeIterator(g, 'call')


def EntryBlockIterator(g: nx.Graph):
  """Iterate over the entry blocks of a graph."""
  for node, data in StatementNodeIterator(g):
    if not g.in_degree(node):
      yield node, data


def ExitBlockIterator(g: nx.Graph):
  """Iterate over the exit blocks of a graph."""
  for node, data in StatementNodeIterator(g):
    if not g.out_degree(node):
      yield node, data


def StatementNeighbors(g: nx.Graph, node: str,
                       flow='control') -> typing.Set[str]:
  """Return the neighboring statements connected by the given flow type."""
  neighbors = set()
  for src, dst in g.out_edges(node):
    if g.edges[src, dst, 0]['flow'] == flow:
      neighbors.add(dst)
  return neighbors


def SuccessorNodes(
    g: nx.DiGraph,
    node: str,
    direction: typing.Optional[
        typing.Callable[[typing.Any, typing.Any], typing.Any]] = None,
    ignored_nodes: typing.Optional[typing.Iterable[str]] = None
) -> typing.List[str]:
  """Find the successor nodes of a node."""
  direction = direction or (lambda src, dst: dst)
  ignored_nodes = ignored_nodes or set()
  real = []
  for src, dst in direction(g.in_edges, g.out_edges)(node):
    if direction(src, dst) not in ignored_nodes:
      real.append(direction(src, dst))
    else:
      # The node is ignored, so skip over it and look for the next
      real += SuccessorNodes(g, direction(src, dst), ignored_nodes, direction)
  return real


def StatementIsSuccessor(g: nx.MultiDiGraph, src: str, dst: str) -> bool:
  """Return True if `dst` is successor to `src`."""
  visited = set()
  q = collections.deque([src])
  while q:
    current = q.popleft()
    if current == dst:
      return True
    visited.add(current)
    for next_node in g.neighbors(current):
      edge_flow = g.edges[current, next_node, 0].get('flow', 'control')
      if not edge_flow == 'control':
        continue
      node_type = g.nodes[next_node].get('type', 'statement')
      if not node_type == 'statement':
        continue
      if next_node in visited:
        continue
      q.append(next_node)
  return False


def InsertFunctionGraph(graph, function_name, function_graphs
                       ) -> typing.Tuple[nx.MultiDiGraph, str, str]:
  """Insert the named function graph to the graph."""
  if function_name not in function_graphs:
    raise ValueError(f"Cannot insert non-existent function `{function_name}`. "
                     f"Available functions: {sorted(function_graphs.keys())}")

  function_graph = copy.deepcopy(function_graphs[function_name])
  graph = nx.compose(graph, function_graph)
  return graph, function_graph.entry_block, function_graph.exit_block


def GetCalledFunctionName(statement) -> typing.Optional[str]:
  """Get the name of a function called in the statement."""
  if 'call ' not in statement:
    return None
  # Try and resolve the call destination.
  _, m_glob, _, _ = inst2vec_preprocess.get_identifiers_from_line(statement)
  if not m_glob:
    return None
  return m_glob[0][1:]  # strip the leading '@' character


def FindCallSites(graph, src, dst):
  """Find the statements in `src` function that call `dst` function."""
  call_sites = []
  for node, data in StatementNodeIterator(graph):
    if data['function'] != src:
      continue
    statement = data.get('original_text', data['text'])
    called_function = GetCalledFunctionName(statement)
    if not called_function:
      continue
    if called_function == dst:
      call_sites.append(node)
  if not call_sites:
    raise ValueError(
        "Unable to find any call statements from `{src}` to `{dst}`")
  return call_sites


def AddInterproceduralCallEdges(
    graph: nx.MultiDiGraph, call_multigraph: nx.MultiDiGraph,
    function_entry_exit_nodes: typing.Dict[str, typing.Tuple[str, str]],
    get_call_site_successor: typing.Callable[[nx.MultiDiGraph, str], str]):
  """Add edges of type "call" between statements to match the call graph."""

  # Since we're not duplicating graphs, we can drop the parallel edges by
  # converting the call graph back to a regular directed graph.
  call_graph = nx.DiGraph(call_multigraph)

  for src, dst in call_graph.edges:
    # Ignore connections to the root node, we have already processed them.
    if src == 'external node':
      continue

    call_sites = FindCallSites(graph, src, dst)

    # Check that the number of call sounds we found matches the expected number
    # from the call graph.
    multigraph_call_count = call_multigraph.number_of_edges(src, dst)
    if len(call_sites) != multigraph_call_count:
      raise ValueError("Call graph contains "
                       f"{humanize.Plural(multigraph_call_count, 'call')} from "
                       f"function `{src}` to `{dst}`, but found "
                       f"{len(call_sites)} call sites in the graph")

    for call_site in call_sites:
      # Lookup the nodes to connect.
      call_entry, call_exit = function_entry_exit_nodes[dst]
      call_site_successor = get_call_site_successor(graph, call_site)
      # Connect the noes.
      graph.add_edge(call_site, call_entry, flow='call')
      graph.add_edge(call_exit, call_site_successor, flow='call')


def GetCallStatementSuccessor(graph: nx.MultiDiGraph, call_site: str):
  """Find the neighboring"""
  call_site_successors = StatementNeighbors(graph, call_site)
  if len(call_site_successors) != 1:
    raise ValueError(
        f"Call statement `{call_site}` should have exactly one successor "
        "statement but found "
        f"{humanize.Plural(len(call_size_successors), 'successor')}: "
        f"`{call_site_successors}`")
  return list(call_site_successors)[0]


class ControlAndDataFlowGraphBuilder(object):

  def __init__(self,
               dataflow: str = 'edges_only',
               preprocess_text: bool = True,
               discard_unknown_statements: bool = False,
               only_add_entry_and_exit_blocks_if_required: bool = True,
               call_edge_returns_to_successor: bool = False):
    """Instantiate a Control and Data Flow Graph (CDFG) builder.

    Args:
      dataflow: One of {none,edges_only,nodes_and_edges}. Determines the type of
        data flow information that is added to control flow graphs. If `none`,
        only control flow is used. if `edges_only`, data flow edges are inserted
        between statements. If `nodes_and_edges`, nodes representing identifiers
        are inserted, and data flow edges flowing between the identifier nodes
        and statements.
      preprocess_text: If true, pre-process the text of statements to discard
        literals, normalise identifiers, etc.
      discard_unknown_statements: Pre-processing can choose to delete statements
        (see inst2vec_preprocess.keep()). In that case, the node can either be
        removed, or the node can be kept but with the text set to `UNK!`. If the
        node is removed, the control flow paths flowing through the node are
        preserved.
      only_add_entry_and_exit_blocks_if_required: If True, insert magic entry
        and exit blocks only if they are required. A magic entry/exit block is
        required only if there exists multiple entrance/exit points in the
        graph.
      call_edge_returns_to_successor: If True, when inserting call edges between
        functions, the edge returning from the called function points to the
        statement after the call statement. If False, the outgoing and return
        edges both point to the call statement.
    """
    self.dataflow = dataflow
    self.preprocess_text = preprocess_text
    self.discard_unknown_statements = discard_unknown_statements
    self.only_add_entry_and_exit_blocks_if_required = (
        only_add_entry_and_exit_blocks_if_required)
    self.call_edge_returns_to_successor = call_edge_returns_to_successor

  def MaybePreprocessStatementText(self, g: nx.Graph) -> None:
    """Replace 'text' statement attributes with inst2vec preprocessed.

    An 'original_text' attribute is added to nodes which stores the unmodified
    text.

    Args:
      g: The graph to pre-process the statement node texts of.
    """
    if not self.preprocess_text:
      return

    for node, data in StatementNodeIterator(g):
      if 'text' not in data:
        raise ValueError(
            f"No `text` attribute for node `{node}` with attributes: {data}")
    lines = [[data['text']] for _, data in StatementNodeIterator(g)]
    preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
    preprocessed_texts = [
        inst2vec_preprocess.PreprocessStatement(x[0] if len(x) else '')
        for x in preprocessed_lines
    ]
    # Pre-processing may through away lines (e.g. 'target datalayout' lines).
    # Keep track of those that have been discarded, to be removed later.
    nodes_to_remove: typing.Set[str] = set()
    edges_to_remove: typing.Set[typing.Tuple[str, str]] = set()
    edges_to_add: typing.Set[typing.Tuple[str, str]] = set()
    for (node, data), text in zip(g.nodes(data=True), preprocessed_texts):
      if text:
        data['original_text'] = data['text']
        data['text'] = text
        data['type'] = 'statement'
      elif self.discard_unknown_statements:
        nodes_to_remove.add(node)
      else:
        data['original_text'] = data['text']
        data['text'] = 'UNK!'
        data['type'] = 'statement'

    # Delete the nodes that have been discarded by preprocessing, and re-connect
    # any edges that flow through the nodes.
    for node in nodes_to_remove:
      in_edges = g.in_edges(node)
      out_edges = g.out_edges(node)
      in_nodes = SuccessorNodes(g,
                                node,
                                ignored_nodes=nodes_to_remove,
                                direction=lambda src, dst: src)
      out_nodes = SuccessorNodes(g,
                                 node,
                                 ignored_nodes=nodes_to_remove,
                                 direction=lambda src, dst: dst)

      for edge in in_edges:
        edges_to_remove.add(edge)
      for edge in out_edges:
        edges_to_remove.add(edge)
      for src, dst in itertools.product(in_nodes, out_nodes):
        edges_to_add.add((src, dst))

    for edge in edges_to_remove:
      g.remove_edge(*edge)
    for node in nodes_to_remove:
      g.remove_node(node)
    for edge in edges_to_add:
      g.add_edge(*edge)

  def MaybeAddSingleEntryBlock(self, g: nx.MultiDiGraph) -> None:
    """Add a magic entry block."""
    entry_blocks = list(EntryBlockIterator(g))
    if not entry_blocks:
      raise ValueError("No entry blocks found in graph!")

    if (self.only_add_entry_and_exit_blocks_if_required and
        len(entry_blocks) == 1):
      entry_block = entry_blocks[0][0]
    else:
      entry_block = f'{g.name}_entry'
      g.add_node(entry_block, name=entry_block, type='magic')
      for node, data in entry_blocks:
        g.add_edge(entry_block, node, flow='control')
    g.entry_block = entry_block

  def MaybeAddSingleExitBlock(self, g: nx.MultiDiGraph) -> None:
    """Add a magic exit block to unite the exit statements of a CFG.

    Args:
      g: The graph to add the exit block to.
    """
    exit_blocks = list(ExitBlockIterator(g))
    if not exit_blocks:
      raise ValueError("No exit blocks found in graph!")

    if (self.only_add_entry_and_exit_blocks_if_required and
        len(exit_blocks) == 1):
      exit_block = exit_blocks[0][0]
    else:
      exit_block = f'{g.name}_exit'
      g.add_node(exit_block, name=exit_block, type='magic')
      # Connect exit blocks.
      for node, data in exit_blocks:
        g.add_edge(node, exit_block, flow='control')
      # Add a dataflow edge out, if there is one.
      for src, dst, data in DataFlowEdgeIterator(g):
        if dst == node:
          g.add_edge(node, exit_block, flow='data')
          break
    g.exit_block = exit_block

  def MaybeAddDataFlowElements(self, g: nx.MultiDiGraph) -> None:
    if self.dataflow == 'none':
      return

    prefix = lambda s: f'{g.name}_{s}'
    unprefix = lambda s: s[len(f'{g.name}_'):]

    # Collect the edges to add so that we don't modify the graph while
    # iterating.
    edges_to_add: typing.List[typing.Tuple[str, str, str]] = []

    for statement, data in StatementNodeIterator(g):
      destination, operands = GetLlvmStatementDestinationAndOperands(
          data['text'])
      if destination:  # Data flow out edge.
        edges_to_add.append(
            (statement, prefix(destination), prefix(destination)))
      for identifier in operands:  # Data flow in edge.
        edges_to_add.append((prefix(identifier), statement, prefix(identifier)))

    for src, dst, identifier in edges_to_add:
      g.add_edge(src, dst, flow='data')
      node = g.nodes[identifier]
      node['type'] = 'identifier'
      node['name'] = unprefix(identifier)

    # Strip the identifier nodes which have no inputs, since they provide no
    # meaningful data 'flow' information.
    nodes_to_remove: typing.List[str] = []
    for node, _ in IdentifierNodeIterator(g):
      if not g.in_degree(node):
        nodes_to_remove.append(node)
    [g.remove_node(node) for node in nodes_to_remove]

    if self.dataflow == 'edges_only':
      # If we only want the dataflow edges, and not the intermediate nodes,
      # then first build a list of all of the incoming and outgoing edges from
      # identifier nodes, then replace them with direct edges and remove the
      # original node and edges.
      data_flow_edges_to_add: typing.List[typing.Tuple[str, str]] = []
      edges_to_remove: typing.List[typing.Tuple[str, str]] = []
      nodes_to_remove: typing.List[str] = []

      for identifier, _ in IdentifierNodeIterator(g):
        nodes_to_remove.append(identifier)
        srcs = list(g.in_edges(identifier))
        dsts = list(g.out_edges(identifier))
        edges_to_remove += list(srcs) + list(dsts)
        for (src, _), (_, dst) in itertools.product(srcs, dsts):
          if StatementIsSuccessor(g, src, dst):
            data_flow_edges_to_add.append((src, dst, unprefix(identifier)))

      # Remove multiple incoming DF edges from the same identifier by keeping
      # only the edge which is closest to the statement (i.e. the most-recent
      # write).
      # TODO(cec): This tramples the DF edge from '%1 = alloca ...' to
      # 'store 1, %1', since both statements are considered writes, so the
      # edge is discarded as a WAW.
      deduped_data_flow_edges = {}
      for src, dst, identifier in data_flow_edges_to_add:
        if (dst, identifier) in deduped_data_flow_edges:
          other_src = deduped_data_flow_edges[(dst, identifier)]
          if StatementIsSuccessor(g, other_src, src):
            deduped_data_flow_edges[(dst, identifier)] = src
        else:
          deduped_data_flow_edges[(dst, identifier)] = src

      [g.remove_edge(*edge) for edge in edges_to_remove]
      [g.remove_node(node) for node in nodes_to_remove]
      [
          g.add_edge(src, dst, identifier=identifier, flow='data')
          for (dst, identifier), src in deduped_data_flow_edges.items()
      ]

  def BuildFromControlFlowGraph(
      self, cfg: llvm_util.LlvmControlFlowGraph) -> nx.DiGraph:
    """Build a CDFG from an LLVM Control Flow Graph.

    Args:
      cfg: The control flow graph to build aCDFG from.
    """
    # Expand the control flow graph to a full flow graph (one block per
    # statement).
    ffg = cfg.BuildFullFlowGraph()

    # Copy the DiGraph to a MultiDiGraph, which is required for the parallel
    # control- and data-flow edges. Also prefix the node and edge names with the
    # name of the graph, so that multiple graphs from the same bytecode file can
    # be composed.
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

    self.MaybeAddDataFlowElements(g)
    self.MaybePreprocessStatementText(g)
    self.MaybeAddSingleEntryBlock(g)
    self.MaybeAddSingleExitBlock(g)
    return g

  def ComposeGraphs(self, function_graphs: typing.List[nx.MultiDiGraph],
                    call_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Combine function-level graphs into a single interprocedural graph.

    Args:
      function_graphs: The function graphs to compose. These are unmodified.
      call_graph: The call graph with which to combine the function graphs.
    """
    function_names = [g.name for g in function_graphs]
    if len(set(function_names)) != len(function_names):
      raise ValueError(
          f"Duplicate function names found: {sorted(function_names)}")

    # Add a function attribute to all nodes to track their originating function.
    for function_graph in function_graphs:
      for _, data in function_graph.nodes(data=True):
        data['function'] = function_graph.name

    function_graph_map = {g.name: g for g in function_graphs}

    # Create the interprocedural graph with a magic root node.
    interprocedural_graph = nx.MultiDiGraph()
    interprocedural_graph.add_node('root', name='root', type='magic')

    # Add each function to the interprocedural graph.
    function_entry_exit_nodes: typing.Dict[str, typing.Tuple[str, str]] = {}

    for _, dst in call_graph.out_edges('external node'):
      interprocedural_graph, function_entry, function_exit = InsertFunctionGraph(
          interprocedural_graph, dst, function_graph_map)
      function_entry_exit_nodes[dst] = (function_entry, function_exit)

      # Connect the newly inserted function to the root node.
      interprocedural_graph.add_edge('root', function_entry, flow='call')

    if self.call_edge_returns_to_successor:
      get_call_site_successor = GetCallStatementSuccessor
    else:
      get_call_site_successor = lambda g, n: n

    AddInterproceduralCallEdges(interprocedural_graph,
                                call_graph,
                                function_entry_exit_nodes,
                                get_call_site_successor=get_call_site_successor)

    return interprocedural_graph

  def Build(self, bytecode: str) -> nx.MultiDiGraph:
    call_graph_dot, cfg_dots = (
        opt_util.DotCallGraphAndControlFlowGraphsFromBytecode(bytecode))
    cfgs = [
        llvm_util.ControlFlowGraphFromDotSource(cfg_dot) for cfg_dot in cfg_dots
    ]
    call_graph = llvm_util.CallGraphFromDotSource(call_graph_dot)
    graphs = [self.BuildFromControlFlowGraph(cfg) for cfg in cfgs]
    return self.ComposeGraphs(graphs, call_graph)


def ToControlFlowGraph(g: nx.MultiDiGraph):
  """Create a new graph which contains only the """
  cfg = nx.MultiDiGraph()

  for node, _ in StatementNodeIterator(g):
    cfg.add_node(node, type='statement')

  for src, dst, _ in ControlFlowEdgeIterator(g):
    cfg.add_edge(src, dst, flow='control')


CreateLabelCallback = typing.Callable[[typing.Dict[str, typing.Any]], str]
KeyOrCallback = typing.Union[str, CreateLabelCallback]
StringOrCallback = typing.Union[str, CreateLabelCallback]


def ToDot(g: nx.Graph,
          statement_label: KeyOrCallback = 'text',
          statement_shape: StringOrCallback = 'box',
          statement_color: StringOrCallback = 'white',
          identifier_label: KeyOrCallback = 'name',
          identifier_shape: StringOrCallback = 'ellipse',
          identifier_color: StringOrCallback = 'white',
          magic_label: KeyOrCallback = 'name',
          magic_shape: StringOrCallback = 'doubleoctagon',
          magic_color: StringOrCallback = 'white',
          control_flow_color: StringOrCallback = 'blue',
          data_flow_label: KeyOrCallback = 'identifier',
          data_flow_color: StringOrCallback = 'red',
          call_edge_color: StringOrCallback = 'green'):
  """Render the dot visualiation of the graph.

  Args:
    node_label: The attribute to use as the node label.
    node_shape: The graphviz shape name to draw nodes in.
  """
  g = g.copy()

  def DataKeyOrCallback(data, key_or_callback) -> str:
    """Return key_or_callback(data) if callable, else data[key_or_callback]."""
    if callable(key_or_callback):
      return key_or_callback(data)
    else:
      return data.get(key_or_callback, '')

  def StringOrCallback(data, string_or_callback) -> str:
    """Return string_or_callback(data) if callable, else string_or_callback."""
    if callable(string_or_callback):
      return string_or_callback(data)
    else:
      return string_or_callback

  # Set node properties

  for _, data in g.nodes(data=True):
    # Add a 'null' attribute to nodes so that they can have empty labels.
    data['null'] = ''
    # Set the node to filled so that their color shows up.
    data['style'] = 'filled'

  for _, _, data in g.edges(data=True):
    # Add a 'null' attribute to edges so that they can have empty labels.
    data['null'] = ''

  for node, data in StatementNodeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, statement_label)}"'
    data['shape'] = StringOrCallback(data, statement_shape)
    data['fillcolor'] = StringOrCallback(data, statement_color)

  for node, data in IdentifierNodeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, identifier_label)}"'
    data['shape'] = StringOrCallback(data, identifier_shape)
    data['fillcolor'] = StringOrCallback(data, identifier_color)

  for node, data in MagicNodeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, magic_label)}"'
    data['shape'] = StringOrCallback(data, magic_shape)
    data['fillcolor'] = StringOrCallback(data, magic_color)

  # Set edge properties.

  for src, dst, data in ControlFlowEdgeIterator(g):
    data['color'] = StringOrCallback(data, control_flow_color)

  for src, dst, data in DataFlowEdgeIterator(g):
    data['label'] = f'"{DataKeyOrCallback(data, data_flow_label)}"'
    data['color'] = StringOrCallback(data, data_flow_color)

  for src, dst, data in CallEdgeIterator(g):
    data['color'] = StringOrCallback(data, call_edge_color)

  # Remove unneeded attributes.
  def DeleteKeys(d, keys):
    for key in keys:
      if key in d:
        del d[key]

  for _, data in g.nodes(data=True):
    DeleteKeys(data, {'original_text', 'type', 'null', 'text', 'name'})
  for _, _, data in g.edges(data=True):
    DeleteKeys(data, {'flow', 'key'})

  buf = io.StringIO()
  nx.drawing.nx_pydot.write_dot(g, buf)
  return buf.getvalue()


def CallGraphToFunctionCallCounts(
    call_graph: nx.MultiDiGraph) -> typing.Dict[str, int]:
  """Build a table of call counts for each function.

  Args:
    call_graph: A call graph, such as produced by LLVM's -dot-callgraph pass.
      See llvm_util.CallGraphFromDotSource().

  Returns:
    A dictionary where each function in the graph is a key, and the value is the
    number of unique call sites for that function. Note this may be zero, in the
    case of library functions.
  """
  # Initialize the call count table with an entry for each function, except
  # the magic "external node" entry produced by LLVM's -dot-callgraph pass.
  function_names = [n for n in call_graph.nodes if n != 'external node']
  call_counts = {n: 0 for n in function_names}
  for src, dst, _ in call_graph.edges:
    if src != 'external node':
      call_counts[dst] += 1
  return call_counts
