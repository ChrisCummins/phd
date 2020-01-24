# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines the ProGraML graph representation.

A ProGraML graph is a directed multigraph which is the union a control flow,
data flow, and call graphs.
"""
import signal
import typing
import warnings

import networkx as nx

from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs import nx_utils
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.llvm2graph import node_encoder
from deeplearning.ml4pl.graphs.llvm2graph.legacy import call_graph as cg
from deeplearning.ml4pl.graphs.llvm2graph.legacy import llvm_statements
from deeplearning.ml4pl.graphs.llvm2graph.legacy.cfg import llvm_util
from labm8.py import app


FLAGS = app.FLAGS


class ProGraMLGraphBuilder(object):
  def __init__(
    self, dataflow: bool = True, preprocess_text: bool = True, opt=None,
  ):
    """Instantiate a Control and Data Flow Graph (CDFG) builder.

    Args:
      dataflow: Determine the type of data flow information that is added to
        control flow graphs. If False, only control flow is used. If True,
        nodes representing identifiers and immediates are inserted, and data
        flow edges flowing between the identifier nodes and statements.
      preprocess_text: If true, pre-process the text of statements to discard
        literals, normalise identifiers, etc.
      opt: The path to LLVM `opt` binary to use to construct control-flow and
        call graphs from. The default uses the opt binary packaged with
        //compilers/llvm:opt.
    """
    self.dataflow = dataflow
    self.preprocess_text = preprocess_text
    self.node_encoder = node_encoder.GraphNodeEncoder()
    self.opt = opt

  def Build(
    self,
    bytecode: str,
    tag_hook: typing.Optional[llvm_util.TagHook] = None,
    timeout_seconds: int = 120,
  ) -> nx.MultiDiGraph:
    """Construct a ProGraML from the given bytecode.

    Args:
      bytecode: The bytecode to construct the graph from.
      tag_hook: An optional object that can tag specific nodes in the graph
                according to some logic.
      timeout_seconds: The maximum number of seconds to run before terminating.

    Returns:
      A networkx graph.
    """
    warnings.warn(
      "Use deeplearning.ml4pl.graphs.llvm2graph.llvm2graph.BuildProgramGraphNetworkX() instead"
    )

    def _RaiseTimoutError(signum, frame):
      del signum
      del frame
      raise TimeoutError(
        f"Graph construction did not complete within {timeout_seconds} seconds"
      )

    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, _RaiseTimoutError)
    signal.alarm(timeout_seconds)

    try:
      return self._Build(bytecode, tag_hook)
    except TimeoutError as e:
      raise e
    finally:
      # Unregister the signal so it won't be triggered
      # if the timeout is not reached.
      signal.signal(signal.SIGALRM, signal.SIG_IGN)

  def _Build(self, bytecode: str, tag_hook: llvm_util.TagHook):
    """Private implementation of Build function."""
    # First construct the control flow graphs using opt.
    (
      call_graph_dot,
      cfg_dots,
    ) = opt_util.DotCallGraphAndControlFlowGraphsFromBytecode(
      bytecode, opt_path=self.opt
    )

    # Then construct the call graph dot using opt.
    call_graph = cg.CallGraphFromDotSource(call_graph_dot)
    # Construct NetworkX control flow graphs from the dot graphs.
    cfgs = [
      llvm_util.ControlFlowGraphFromDotSource(cfg_dot, tag_hook=tag_hook)
      for cfg_dot in cfg_dots
    ]

    # Add data flow elements to control flow graphs.
    graphs = [self.CreateControlAndDataFlowUnion(cfg) for cfg in cfgs]
    # Finally, compose the per-function graphs into a whole-module graph.
    return self.ComposeGraphs(graphs, call_graph)

  def CreateControlAndDataFlowUnion(
    self, cfg: llvm_util.LlvmControlFlowGraph
  ) -> nx.MultiDiGraph:
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
    # While doing this, prefix the node and edge names with the name of the
    # graph so that multiple graphs from the same bytecode file can be composed.
    g = nx.MultiDiGraph()
    g.graph["name"] = ffg.name
    for node, data in ffg.nodes(data=True):
      g.add_node(
        f"{g.name}_{node}",
        text=data["text"],
        name=data["name"],
        type=programl_pb2.Node.STATEMENT,
        function=g.name,
      )

    for src, dst, data in ffg.edges(data=True):
      g.add_edge(
        f"{g.name}_{src}",
        f"{g.name}_{dst}",
        position=data["position"],
        flow=programl_pb2.Edge.CONTROL,
      )

    # Record the graph entry block.
    g.graph["entry_block"] = f"{g.name}_{ffg.entry_block}"

    self.node_encoder.EncodeNodes(g)

    self.AddDataFlowElements(g, ffg.tag_hook)
    # self.MaybePreprocessStatementText(g)
    self.MaybeAddSingleExitBlock(g)
    return g

  def MaybeAddSingleExitBlock(self, g: nx.MultiDiGraph) -> None:
    """Add a magic exit block to unite the exit statements of a CFG.

    Args:
      g: The graph to add the exit block to.
    """
    exit_blocks = list(nx_utils.ExitBlockIterator(g))
    if not exit_blocks:
      raise ValueError("No exit blocks found in graph!")

    if len(exit_blocks) == 1:
      exit_block = exit_blocks[0][0]
    else:
      exit_block = f"{g.name}_exit"
      g.add_node(
        exit_block,
        name=exit_block,
        type=programl_pb2.Node.STATEMENT,
        text="",
        preprocessed_text="",
        function=None,
        x=[self.node_encoder.dictionary["!MAGIC"]],
        y=[],
      )
      # Connect exit blocks.
      for node, data in exit_blocks:
        g.add_edge(node, exit_block, flow=programl_pb2.Edge.CONTROL, position=0)
      # Add a dataflow edge out, if there is one.
      for src, dst, data in nx_utils.DataFlowEdgeIterator(g):
        if dst == node:
          g.add_edge(node, exit_block, flow=programl_pb2.Edge.DATA, position=0)
          break
    g.graph["exit_block"] = exit_block

  def AddDataFlowElements(
    self, g: nx.MultiDiGraph, tag_hook: typing.Optional[llvm_util.TagHook]
  ) -> None:
    if self.dataflow == "none":
      return

    prefix = lambda s: f"{g.name}_{s}"
    unprefix = lambda s: s[len(f"{g.name}_") :]

    # Collect the edges to add so that we don't modify the graph while
    # iterating.
    edges_to_add: typing.List[typing.Tuple[str, str, str, int]] = []

    for statement, data in nx_utils.StatementNodeIterator(g):
      # TODO(github.com/ChrisCummins/ProGraML/issues/9): Separate !IDENTIFIER
      # and !IMMEDIATE uses.
      def_, uses = llvm_statements.GetLlvmStatementDefAndUses(data["text"])
      if def_:  # Data flow out edge.
        def_name = f"{prefix(def_)}_operand"
        edges_to_add.append(
          (statement, def_name, def_name, 0, def_, data, "def")
        )
      for position, identifier in enumerate(uses):  # Data flow in edge.
        identifier_name = f"{prefix(identifier)}_operand"
        edges_to_add.append(
          (
            identifier_name,
            statement,
            identifier_name,
            position,
            identifier,
            data,
            "use",
          )
        )

    for (
      src,
      dst,
      identifier,
      position,
      name,
      original_node,
      dtype,
    ) in edges_to_add:
      g.add_edge(src, dst, flow=programl_pb2.Edge.DATA, position=position)
      node = g.nodes[identifier]
      # The node may have been implicitly created. Set its attributes now.
      # TODO(github.com/ChrisCummins/ProGraML/issues/9): Separate !IDENTIFIER
      # and !IMMEDIATE nodes.
      node["type"] = programl_pb2.Node.IDENTIFIER
      node["name"] = name
      node["text"] = name
      node["preprocessed_text"] = "!IDENTIFIER"
      node["x"] = [self.node_encoder.dictionary["!IDENTIFIER"]]
      node["y"] = []

      if tag_hook is not None:
        other_attrs = tag_hook.OnIdentifier(original_node, node, dtype) or {}
        for attrname, attrval in other_attrs.items():
          node[attrname] = attrval

  def DictionaryLookup(self, statement: str) -> int:
    if statement in self.node_encoder.dictionary:
      return self.node_encoder.dictionary[statement]
    else:
      return self.node_encoder.dictionary["!UNK"]

  def ComposeGraphs(
    self, function_graphs: typing.List[nx.DiGraph], call_graph: nx.MultiDiGraph,
  ) -> nx.MultiDiGraph:
    """Combine function-level graphs into a single inter-procedural graph.

    Args:
      function_graphs: The function graphs to compose. These are unmodified.
      call_graph: The call graph with which to combine the function graphs.
    """
    function_names = [g.name for g in function_graphs]
    if len(set(function_names)) != len(function_names):
      raise ValueError(
        f"Duplicate function names found: {sorted(function_names)}"
      )

    if "external node" not in call_graph.nodes():
      raise ValueError("Call graph missing `external node` root")

    # Add a function attribute to all nodes to track their originating function.
    for function_graph in function_graphs:
      for _, data in function_graph.nodes(data=True):
        data["function"] = function_graph.name

    function_graph_map = {g.name: g for g in function_graphs}

    # Create the inter-procedural graph with a magic root node.
    interprocedural_graph = nx.MultiDiGraph()

    # Add the unused graph-level attributes.
    interprocedural_graph.graph["x"] = []
    interprocedural_graph.graph["y"] = []

    interprocedural_graph.add_node(
      0,
      type=programl_pb2.Node.STATEMENT,
      text="root",
      preprocessed_text="",
      function=None,
      x=[self.node_encoder.dictionary["!MAGIC"]],
      y=[],
    )

    # Add each function to the inter-procedural graph.
    function_entry_exit_nodes: typing.Dict[str, typing.Tuple[int, int]] = {}

    for _, dst in call_graph.out_edges("external node"):
      (
        interprocedural_graph,
        function_entry,
        function_exit,
      ) = InsertFunctionGraph(
        interprocedural_graph,
        dst,
        function_graph_map,
        self.node_encoder.dictionary,
      )
      function_entry_exit_nodes[dst] = (function_entry, function_exit)

      # Connect the newly inserted function to the root node.
      interprocedural_graph.add_edge(
        0, function_entry, flow=programl_pb2.Edge.CALL, position=0
      )

    AddInterproceduralCallEdges(
      interprocedural_graph, call_graph, function_entry_exit_nodes,
    )

    return interprocedural_graph


def ToControlFlowGraph(g: nx.MultiDiGraph):
  """Create a new graph with only the statements and control flow edges."""
  # CFGs cannot have parallel edges, so we use only a DiGraph rather than
  # MultiDiGraph.
  cfg = nx.DiGraph()

  for node, data in nx_utils.StatementNodeIterator(g):
    cfg.add_node(node, **data)

  for src, dst, data in nx_utils.ControlFlowEdgeIterator(g):
    cfg.add_edge(src, dst, **data)


def SerializeToStatementList(
  g: nx.MultiDiGraph, root: str = "root"
) -> typing.Iterable[str]:
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
      for _, dst, flow in g.out_edges(
        node, data="flow", default=programl_pb2.Edge.CONTROL
      ):
        if flow == programl_pb2.Edge.CONTROL:
          if dst not in visited_statements:
            stack.append(dst)
        elif flow == programl_pb2.Edge.CALL:
          if dst not in visited_functions:
            functions.append(dst)


def MakeUndefinedFunctionGraph(
  function_name: str, dictionary
) -> nx.MultiDiGraph:
  """Create an empty function with the given name.

  Use this to create function graphs for undefined functions. The generated
  functions consist only of an entry and exit block, with a control edge
  between them.
  """
  g = nx.MultiDiGraph()

  g.name = function_name
  g.graph["entry_block"] = f"{function_name}_entry"
  g.graph["exit_block"] = f"{function_name}_exit"

  g.add_node(
    g.graph["entry_block"],
    type=programl_pb2.Node.STATEMENT,
    function=function_name,
    preprocessed_text="!UNK",
    text=g.graph["entry_block"],
    x=[dictionary["!UNK"]],
    y=[],
  )
  g.add_node(
    g.graph["exit_block"],
    type=programl_pb2.Node.STATEMENT,
    function=function_name,
    preprocessed_text="!UNK",
    text=g.graph["exit_block"],
    x=[dictionary["!UNK"]],
    y=[],
  )
  g.add_edge(
    g.graph["entry_block"],
    g.graph["exit_block"],
    function=function_name,
    flow=programl_pb2.Edge.CONTROL,
    position=0,
  )

  return g


def InsertFunctionGraph(
  graph, function_name, function_graphs, dictionary
) -> typing.Tuple[nx.MultiDiGraph, str, str]:
  """Insert the named function graph to the graph."""
  if function_name not in function_graphs:
    function_graphs[function_name] = MakeUndefinedFunctionGraph(
      function_name, dictionary
    )

  function_graph = function_graphs[function_name]

  # Translate string node names to integer values.
  node_map = {
    x: i
    for i, x in enumerate(function_graph.nodes, start=graph.number_of_nodes())
  }

  # Recreate the nodes.
  for node, data in function_graph.nodes(data=True):
    graph.add_node(
      node_map[node],
      function=function_name,
      type=data["type"],
      text=data["text"],
      preprocessed_text=data["preprocessed_text"],
      x=data["x"],
      y=data["y"],
    )

  # Recreate the edges.
  for src, dst, data in function_graph.edges(data=True):
    graph.add_edge(
      node_map[src], node_map[dst], position=data["position"], flow=data["flow"]
    )

  return (
    graph,
    node_map[function_graph.graph["entry_block"]],
    node_map[function_graph.graph["exit_block"]],
  )


def AddInterproceduralCallEdges(
  graph: nx.MultiDiGraph,
  call_multigraph: nx.MultiDiGraph,
  function_entry_exit_nodes: typing.Dict[str, typing.Tuple[int, int]],
) -> None:
  """Add call edges between procedures to match the call graph.

  Args:
    graph: The disconnected per-function graphs.
    call_multigraph: A call graph with parallel edges to indicate multiple calls
      from the same function.
    function_entry_exit_nodes: A mapping from function name to a tuple of entry
      and exit statements.
  """
  # Drop the parallel edges by converting the call graph back to a regular
  # directed graph. Iterating over the edges in this graph then provides the
  # functions that need connecting, while the multigraph tells us how many
  # connections to expect.
  call_graph = nx.DiGraph(call_multigraph)

  for src, dst in call_graph.edges:
    # Ignore connections to the root node, we have already processed them.
    if src == "external node":
      continue

    call_sites = llvm_statements.FindCallSites(graph, src, dst)

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
      # Connect the nodes.
      graph.add_edge(
        call_site, call_entry, flow=programl_pb2.Edge.CALL, position=0
      )
      graph.add_edge(
        call_exit, call_site, flow=programl_pb2.Edge.CALL, position=0
      )
