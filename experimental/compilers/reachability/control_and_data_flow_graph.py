"""This module defines the Control and Data Flow Graph (CDFG).

CDFGs are directed multigraphs in which nodes represent single LLVM
instructions, and edges show control and data flow. They are constructed from an
LLVM module.
"""
import itertools

import io
import networkx as nx
import typing

from deeplearning.ncc.inst2vec import inst2vec_preprocess
from experimental.compilers.reachability import llvm_util
from labm8 import app


FLAGS = app.FLAGS


def GetLlvmStatementDestinationAndOperands(
    statement: str) -> typing.Tuple[str, typing.List[str]]:
  """Get the destination identifier for an LLVM statement (if any), and a list
  of operand identifiers (if any).
  """
  destination, operands = '', []

  if '=' in statement:
    destination, statement = statement.split('=')

  m_loc, m_glob, m_label, m_label2 = inst2vec_preprocess.get_identifiers_from_line(statement)
  operands += m_loc + m_glob + m_label + m_label2

  # Store is a special case because it doesn't have an LHS, but writes to one
  # of its operands.
  if statement.startswith('store '):
    if len(operands) == 1:
      destination = operands[0]
      operands = []
    elif len(operands) == 2:
      destination = operands[0]
      operands = [operands[1]]
    else:
      raise ValueError(f"Unable to extract operands from store statement: `{statement}`")

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


def StatementNeighbors(g: nx.Graph, node: str, flow='control') -> typing.Set[str]:
  """Return the neighboring statements connected by the given flow type."""
  neighbors = set()
  for src, dst in g.out_edges(node):
    if g.edges[src, dst, 0]['flow'] == flow:
      neighbors.add(dst)
  return neighbors


def SuccessorNodes(g: nx.DiGraph, node: str,
                   direction: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None,
                   ignored_nodes: typing.Optional[typing.Iterable[str]] = None):
  """Find the successor nodes of a node."""
  direction = direction or (lambda src, dst: dst  )
  ignored_nodes = ignored_nodes or set()
  real = []
  for src, dst in direction(g.in_edges, g.out_edges)(node):
    if direction(src, dst) not in ignored_nodes:
      real.append(direction(src, dst))
    else:
      # The node is ignored, so skip over it and look for the next
      real += SuccessorNodes(g, direction(src, dst), ignored_nodes, direction)
  return real


class ControlAndDataFlowGraphBuilder(object):

  def __init__(self, dataflow: str = 'edges_only',
               preprocess_text: bool = True,
               discard_unknown_statements: bool = False,
               add_entry_block: bool=True,
               add_exit_block: bool=True):
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
      add_entry_block:
    """
    self.dataflow = dataflow
    self.preprocess_text = preprocess_text
    self.discard_unknown_statements = discard_unknown_statements
    self.add_entry_block = add_entry_block
    self.add_exit_block = add_exit_block

    self.cfg_count = 0

  def PreprocessStatementText(self, g: nx.Graph) -> None:
    """Replace the 'text' attribute of statement nodes with inst2vec preprocessed.

    The original text is added to an 'original_text' property of nodes.

    Args:
      g: The graph to pre-process the statement node texts of.
    """
    if not self.preprocess_text:
      return

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
      in_nodes = SuccessorNodes(g, node, ignored_nodes=nodes_to_remove,
                                direction=lambda src, dst: src)
      out_nodes = SuccessorNodes(g, node, ignored_nodes=nodes_to_remove,
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

  def AddEntryBlock(self, g: nx.MultiDiGraph) -> None:
    """Add a magic entry block."""
    if not self.add_entry_block:
      return
    entry_block = f'{g.name}_entry'
    g.add_node(entry_block, name=entry_block, type='magic')
    for node, data in EntryBlockIterator(g):
      g.add_edge(entry_block, node, flow='control')

  def AddExitBlock(self, g: nx.MultiDiGraph) -> None:
    """Add a magic exit block."""
    if not self.add_exit_block:
      return
    exit_block = f'{g.name}_exit'
    g.exit_block = exit_block
    g.add_node(exit_block, name=exit_block, type='magic')
    for node, data in ExitBlockIterator(g):
      g.add_edge(node, exit_block, flow='control')
      # Add a dataflow edge out, if there is one.
      for src, dst, data in DataFlowEdgeIterator(g):
        if dst == node:
          g.add_edge(node, exit_block, flow='data')
          break

  def AddDataFlowToControlFlowGraph(self, g: nx.DiGraph) -> None:
    if self.dataflow != "none":
      # Collect the edges to add so that we don't modify the graph while
      # iterating.
      edges_to_add: typing.List[typing.Tuple[str, str, str]] = []
      prefix = lambda s: f'{g.name}_{s}'
      unprefix = lambda s: s[len(f'{g.name}_'):]

      for statement, data in StatementNodeIterator(g):
        destination, operands = GetLlvmStatementDestinationAndOperands(data['text'])
        if destination:  # Data flow out edge.
          edges_to_add.append((statement, prefix(destination), prefix(destination)))
        for identifier in operands:  # Data flow in edge.
          edges_to_add.append((prefix(identifier), statement, prefix(identifier)))

      for src, dst, identifier in edges_to_add:
        g.add_edge(src, dst, flow='data')
        node = g.nodes[identifier]
        node['type'] = 'identifier'
        node['name'] = unprefix(identifier)

      if self.dataflow == "edges_only":
        # If we only want the dataflow edges, and not the intermediate nodes,
        # then first build a list of all of the incoming and outgoing edges from
        # identifier nodes, then replace them with direct edges and remove the
        # original node and edges.
        edges_to_add: typing.List[typing.Tuple[str, str]] = []
        edges_to_remove: typing.List[typing.Tuple[str, str]] = []
        nodes_to_remove: typing.List[str] = []
        for node, data in IdentifierNodeIterator(g):
          nodes_to_remove.append(node)
          srcs = list(g.in_edges(node))
          dsts = list(g.out_edges(node))
          edges_to_remove += list(srcs) + list(dsts)
          for src, dst in itertools.product(srcs, dsts):
            edges_to_add.append((src[0], dst[1]))

        for edge in edges_to_remove:
          g.remove_edge(*edge)
        for node in nodes_to_remove:
          g.remove_node(node)
        for edge in edges_to_add:
          g.add_edge(*edge, flow='data')

  def BuildFromControlFlowGraph(self, cfg: llvm_util.LlvmControlFlowGraph) -> nx.DiGraph:
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

    self.AddDataFlowToControlFlowGraph(g)
    self.PreprocessStatementText(g)
    self.AddEntryBlock(g)
    self.AddExitBlock(g)
    return g

  def ComposeGraphs(self, graphs: typing.List[nx.MultiDiGraph]) -> nx.MultiDiGraph:
    """Combine per-functions graphs into a single whole-module graph."""
    graph_names = [g.name for g in graphs]
    entry_blocks = {g.name: g.entry_block for g in graphs}
    exit_blocks = {g.name: g.exit_block for g in graphs}

    # Compose the graphs into a single big one.
    g = nx.MultiDiGraph()
    for graph in graphs:
      g = nx.compose(g, graph)

    # Connect the call statements.
    edges_to_remove = set()
    edges_to_add = set()
    for node, data in StatementNodeIterator(g):
      statement = data.get('original_text', data['text'])
      if 'call ' in statement:
        # Try and resolve the call destination.
        _, m_glob, _, _ = inst2vec_preprocess.get_identifiers_from_line(statement)
        if not m_glob:
          continue
        destination = m_glob[0][1:]
        if destination not in graph_names:
          continue

        edges_to_add.add((node, entry_blocks[destination]))
        for edge in g.out_edges(node):
          data = g.edges[(edge[0], edge[1], 0)]
          if data['flow'] == 'control':
            edges_to_remove.add(edge)
            edge = (exit_blocks[destination], edge[1])
            edges_to_add.add(edge)

    for edge in edges_to_remove:
      g.remove_edge(*edge)
    for edge in edges_to_add:
      assert edge[0] in g.nodes
      assert edge[1] in g.nodes
      g.add_edge(*edge, flow='control')

    root = 'root'
    g.add_node(root, name='root', type='magic')
    for entry_block in entry_blocks.values():
      if g.in_degree(entry_block) == 0:
        g.add_edge(root, entry_block, flow='control')

    return g

  def BuildFromControlFlowGraphs(
      self, cfgs: typing.List[llvm_util.LlvmControlFlowGraph]) -> nx.DiGraph:
    graphs = [self.BuildFromControlFlowGraph(cfg) for cfg in cfgs]
    return self.ComposeGraphs(graphs)

  def Build(self, bytecode: str) -> nx.MultiDiGraph:
    cfgs = list(llvm_util.ControlFlowGraphsFromBytecodes([bytecode]))
    return self.BuildFromControlFlowGraphs(cfgs)


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
          control_flow_color: StringOrCallback = 'black',
          data_flow_color: StringOrCallback = 'blue'):
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
      return data[key_or_callback]

  def StringOrCallback(data, string_or_callback) -> str:
    """Return string_or_callback(data) if callable, else string_or_callback."""
    if callable(string_or_callback):
      return string_or_callback(data)
    else:
      return string_or_callback

  # Set node properties

  for (_, data) in g.nodes(data=True):
    # Add a 'null' attribute to nodes so that they can have empty labels.
    data['null'] = ''
    # Set the node to filled so that their color shows up.
    data['style'] = 'filled'

  for node, data in StatementNodeIterator(g):
    data['label'] = DataKeyOrCallback(data, statement_label)
    data['shape'] = StringOrCallback(data, statement_shape)
    data['fillcolor'] = StringOrCallback(data, statement_color)

  for node, data in IdentifierNodeIterator(g):
    data['label'] = DataKeyOrCallback(data, identifier_label)
    data['shape'] = StringOrCallback(data, identifier_shape)
    data['fillcolor'] = StringOrCallback(data, identifier_color)

  for node, data in MagicNodeIterator(g):
    data['label'] = DataKeyOrCallback(data, magic_label)
    data['shape'] = StringOrCallback(data, magic_shape)
    data['fillcolor'] = StringOrCallback(data, magic_color)

  # Set edge properties.

  for src, dst, data in ControlFlowEdgeIterator(g):
    data['color'] = StringOrCallback(data, data_flow_color)

  for src, dst, data in DataFlowEdgeIterator(g):
    data['color'] = StringOrCallback(data, control_flow_color)

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
