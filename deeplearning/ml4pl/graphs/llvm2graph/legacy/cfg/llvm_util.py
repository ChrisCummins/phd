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
"""Utility code for creating CFGs and FFGs from LLVM bytecodes."""
import multiprocessing
import re
import typing

import networkx as nx
import pydot
import pyparsing

from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs.llvm2graph.legacy.cfg import (
  control_flow_graph as cfg,
)
from labm8.py import app
from labm8.py import humanize


FLAGS = app.FLAGS


def NodeAttributesToBasicBlock(
  node_attributes: typing.Dict[str, str]
) -> typing.Dict[str, str]:
  """Get the basic block graph node attributes from a dot graph node.

  Args:
    node_attributes: The dictionary of dot graph information.

  Returns:
    A dictionary of node attributes.
  """
  label = node_attributes.get("label", "")
  if not label.startswith('"{'):
    raise ValueError(f"Unhandled label: '{label}'")
  # Lines are separated using '\l' in the dot label.
  lines = label.split("\l")
  return {
    # The name is in the first line.
    "name": lines[0][len('"{') :].split(":")[0],
    # All other lines except the last are either blank or contain instructions.
    "text": "\n".join(x.lstrip() for x in lines[1:-1] if x.lstrip()),
  }


def SplitInstructionsInBasicBlock(text: str) -> typing.List[str]:
  """Split the individual instructions from a basic block.

  Args:
    text: The text of the basic block.

  Returns:
    A list of lines.
  """
  lines = text.split("\n")
  # LLVM will wrap long lines by inserting an ellipsis at the start of the line.
  # Undo this.
  for i in range(len(lines) - 1, 0, -1):
    if lines[i].startswith("... "):
      lines[i - 1] += lines[i][len("...") :]
      lines[i] = None
  return [line for line in lines if line]


class TagHook(object):
  """An object that is called while parsing an LLVM dot graph to a CFG. Used, e.g., 
  for annotation. """

  def OnGraphBegin(self, dot: pydot.Dot):
    """Called upon first encounter of a dot graph."""
    pass

  def OnNode(self, node: pydot.Node) -> typing.Dict[str, typing.Any]:
    """Called when a node is encountered. Returns additional attributes to encountered node."""
    pass

  def OnInstruction(
    self, node_attrs: typing.Dict[str, typing.Any], instruction: str
  ) -> typing.Dict[str, typing.Any]:
    """Called when an instruction is created in full-flow graphs. 
    Returns additional attributes to encountered node."""
    pass

  def OnIdentifier(
    self,
    stmt_node: typing.Dict[str, typing.Any],
    identifier_node: typing.Dict[str, typing.Any],
    definition_type: str,
  ) -> typing.Dict[str, typing.Any]:
    """Called when an identifier is created from an instruction in a CDFG.

    Args:
      definition_type: can either be "def" or "use".

    Returns:
      Additional attributes to encountered node.
    """
    pass


class LlvmControlFlowGraph(cfg.ControlFlowGraph):
  """A subclass of the generic control flow graph for LLVM CFGs.

  Each node in an LlvmControlFlowGraph has an additional "text" attribute which
  contains the LLVM instructions for the basic block as a string.
  """

  def __init__(
    self, *args, tag_hook: typing.Optional[TagHook] = None, **kwargs
  ):
    super(LlvmControlFlowGraph, self).__init__(*args, **kwargs)
    self.tag_hook = tag_hook

  def BuildFullFlowGraph(
    self,
    remove_unconditional_branch_statements: bool = False,
    remove_labels_in_branch_statements: bool = False,
  ) -> "LlvmControlFlowGraph":
    """Build a full program flow graph from the Control Flow Graph.

    This expands the control flow graph so that every node contains a single
    LLVM instruction. The "text" attribute of nodes contains the instruction,
    and the "basic_block" attribute contains an integer ID for the basic block
    that the statement came from.

    Node and edge indices cannot be compared between the original graph and
    the flow graph.

    Returns:
      A new LlvmControlFlowGraph in which every node contains a single
      instruction.

    Raises:
      MalformedControlFlowGraphError: In case the CFG is not valid.
    """
    self.ValidateControlFlowGraph(strict=False)

    # Create a new graph.
    sig = LlvmControlFlowGraph(name=self.graph["name"], tag_hook=self.tag_hook)

    # A global node count used to assign unique IDs to nodes in the new graph.
    sig_node_count = 0

    # When expanding the CFG to have a single block for every instruction, we
    # replace a single node with a contiguous run of nodes. We construct a map
    # of self.node IDs to (start,end) tuples of g.node IDs, allowing us to
    # translate node IDs for adding edges.
    class NodeTranslationMapValue(typing.NamedTuple):
      """A [start,end] range of graph node IDs."""

      start: int
      end: int

    node_translation_map: typing.Dict[int, NodeTranslationMapValue] = {}

    # Iterate through all blocks in the source graph.
    for node, data in self.nodes(data=True):
      instructions = SplitInstructionsInBasicBlock(data["text"])
      last_instruction = len(instructions) - 1

      # Split a block into a list of instructions and create a new destination
      # node for each instruction.
      for block_instruction_count, instruction in enumerate(instructions):
        # The ID of the new node is the global node count, plus the offset into
        # the basic block instructions.
        new_node_id = sig_node_count + block_instruction_count
        # The new node name is a concatenation of the basic block name and the
        # instruction count.
        new_node_name = f"{data['name']}.{block_instruction_count}"

        if self.tag_hook:
          other_attrs = self.tag_hook.OnInstruction(data, instruction) or {}
        else:
          other_attrs = {}

        # Add a new node to the graph for the instruction.
        if (
          block_instruction_count == last_instruction
          and instruction.startswith("br ")
        ):
          # Branches can either be conditional, e.g.
          #     br il %6, label %7, label %8
          # or unconditional, e.g.
          #     br label %9
          # Unconditional branches can be skipped - they contain no useful
          # information. Conditional branches can have the labels stripped.
          branch_instruction_components = instruction.split(", ")

          if remove_labels_in_branch_statements:
            branch_text = instruction
          else:
            branch_text = branch_instruction_components[0]

          if len(branch_instruction_components) == 1:
            # Unconditional branches may ignored - they provide no meaningful
            # information beyond what the edge already includes.
            if remove_unconditional_branch_statements:
              block_instruction_count -= 1
            else:
              sig.add_node(
                new_node_id,
                name=new_node_name,
                text=branch_text,
                basic_block=node,
                **other_attrs,
              )
          else:
            sig.add_node(
              new_node_id,
              name=new_node_name,
              text=branch_text,
              basic_block=node,
              **other_attrs,
            )
        else:
          # Regular instruction to add.
          sig.add_node(
            new_node_id,
            name=new_node_name,
            text=instruction,
            basic_block=node,
            **other_attrs,
          )

      # Add an entry to the node translation map for the start and end nodes
      # of this basic block.
      node_translation_map[node] = NodeTranslationMapValue(
        start=sig_node_count, end=sig_node_count + block_instruction_count
      )

      # Create edges between the sequential instruction nodes we just added
      # to the graph.
      [
        sig.add_edge(i, i + 1, position=0)
        for i in range(sig_node_count, sig_node_count + block_instruction_count)
      ]

      # Update the global node count to be the value of the next unused node ID.
      sig_node_count += block_instruction_count + 1

    # Iterate through the edges in the source graph, translating their IDs to
    # IDs in the new graph using the node_translation_map.
    for src, dst, position in self.edges(data="position"):
      new_src = node_translation_map[src].end
      new_dst = node_translation_map[dst].start
      sig.add_edge(new_src, new_dst, position=position)

    # Set the "entry" and "exit" blocks.
    new_entry_block = node_translation_map[self.entry_block].start
    sig.nodes[new_entry_block]["entry"] = True

    for block in self.exit_blocks:
      new_exit_block = node_translation_map[block].end
      sig.nodes[new_exit_block]["exit"] = True

    return sig.ValidateControlFlowGraph(strict=False)

  def ValidateControlFlowGraph(
    self, strict: bool = True
  ) -> "LlvmControlFlowGraph":
    """Validate the control flow graph."""
    super(LlvmControlFlowGraph, self).ValidateControlFlowGraph(strict=strict)

    # Check that each basic block has a text section.
    for _, data in self.nodes(data=True):
      if not data.get("text"):
        raise cfg.MalformedControlFlowGraphError(
          f"Missing 'text' attribute from node '{data}'"
        )

    return self


# A regular expression to match the function name of a CFG dot generated by
# opt.
_DOT_CFG_FUNCTION_NAME_RE = re.compile(r"\".* for '(.+)' function\"")


def ControlFlowGraphFromDotSource(
  dot_source: str,
  remove_blocks_without_predecessors: bool = True,
  tag_hook: typing.Optional[TagHook] = None,
) -> LlvmControlFlowGraph:
  """Create a control flow graph from an LLVM-generated dot file.

  The control flow graph generated from the dot source is not guaranteed to
  be valid. That is, it may contain fusible basic blocks. This can happen if
  the creating the graph from unoptimized bytecode. To disable this generate
  the bytecode with optimizations enabled, e.g. clang -emit-llvm -O3 -S ...

  Args:
    dot_source: The dot source generated by the LLVM -dot-cfg pass.
    remove_blocks_without_predecessors: If true, remove CFG blocks without
      predecessors (except for the entry block). This is similar to the
      -simplify-cfg opt pass.
    tag_hook: An optional object that can tag specific statements in the CFG
              according to some logic.

  Returns:
    A ControlFlowGraph instance.

  Raises:
    pyparsing.ParseException: If dotfile could not be parsed.
    ValueError: If dotfile could not be interpretted / is malformed.
  """
  try:
    parsed_dots = pydot.graph_from_dot_data(dot_source)
  except TypeError as e:
    raise pyparsing.ParseException("Failed to parse dot source") from e

  if len(parsed_dots) != 1:
    raise ValueError(f"Expected 1 Dot in source, found {len(parsed_dots)}")

  dot = parsed_dots[0]

  if tag_hook:
    tag_hook.OnGraphBegin(dot)

  function_name_match = re.match(_DOT_CFG_FUNCTION_NAME_RE, dot.get_name())
  if not function_name_match:
    raise ValueError(f"Could not interpret graph name '{dot.get_name()}'")

  # Create the ControlFlowGraph instance.
  graph = LlvmControlFlowGraph(
    name=function_name_match.group(1), tag_hook=tag_hook
  )

  # Opt names nodes like 'Node0x7f86c670c590'. We discard those names and assign
  # nodes simple integer names.
  # Create the nodes and build a map from node names to indices.
  node_name_to_index_map = {}
  for i, node in enumerate(dot.get_nodes()):
    if node.get_name() in node_name_to_index_map:
      raise ValueError(f"Duplicate node name: '{node.get_name()}'")
    node_name_to_index_map[node.get_name()] = i
    if tag_hook:
      other_attrs = tag_hook.OnNode(node) or {}
    else:
      other_attrs = {}
    graph.add_node(
      i, **NodeAttributesToBasicBlock(node.get_attributes()), **other_attrs
    )

  # Create edges and encode their position. The position is an integer starting
  # at zero and increasing for each outgoing edge, e.g. a switch with `n` cases
  # will have 0..(n-1) unique positions.
  for edge in dot.get_edges():
    # In the dot file, an edge looks like this:
    #     Node0x7f86c670c590:s0 -> Node0x7f86c65001a0;
    src_components = edge.get_source().split(":")
    if len(src_components) > 2:
      raise ValueError(
        f"Cannot interpret edge source name `{edge.get_source()}`"
      )
    elif len(src_components) == 2:
      # Case: Node0x7f87aaf14520:s0
      src_name, position_name = src_components
      position = int(position_name[1:])
    else:
      # Case: Node0x7f87aaf14520
      src_name, position = src_components[0], 0

    src = node_name_to_index_map[src_name]
    dst = node_name_to_index_map[edge.get_destination()]
    graph.add_edge(src, dst, position=position)

  # Optionally remove blocks without predecessors (except the entry block).
  # This emulates the behaviour of the -simplify-cfg opt pass.
  if remove_blocks_without_predecessors:
    removed_blocks_count = 0
    changed = True
    while changed:
      changed = False
      nodes_to_remove = []
      for i, node in enumerate(graph.nodes()):
        if i and graph.in_degree(node) == 0:
          nodes_to_remove.append(node)

      removed_blocks_count += len(nodes_to_remove)
      for node in nodes_to_remove:
        graph.remove_node(node)
        changed = True

    if removed_blocks_count:
      app.Log(
        1,
        "Removed %s without predecessors from `%s`, " "%d blocks remaining",
        humanize.Plural(removed_blocks_count, "block"),
        graph.name,
        graph.number_of_nodes(),
      )

      # Rename nodes to retain a contiguous numeric sequence.
      node_rename_map = {
        oldname: newname for newname, oldname in enumerate(graph.nodes())
      }
      nx.relabel_nodes(graph, node_rename_map, copy=False)

  # The first block is always the function entry.
  graph.nodes[0]["entry"] = True

  # Mark the exit node.
  exit_count = 0
  for node, data in graph.nodes(data=True):
    if graph.out_degree(node) == 0:
      exit_count += 1
      data["exit"] = True

  # CFGs may have multiple exit blocks, e.g. unreachables, exceptions, etc.
  # However, they should have at least one block.
  if exit_count < 1:
    raise ValueError(f"Function `{graph.name}` has no exit blocks")

  return graph


class DotControlFlowGraphsFromBytecodeError(ValueError):
  """An error raised processing a bytecode file."""

  def __init__(self, bytecode: str, error: Exception):
    self.input = bytecode
    self.error = error


class ControlFlowGraphFromDotSourceError(ValueError):
  """An error raised processing a dot source."""

  def __init__(self, dot: str, error: Exception):
    self.input = dot
    self.error = error


def _DotControlFlowGraphsFromBytecodeToQueue(
  bytecode: str, queue: multiprocessing.Queue
) -> None:
  """Process a bytecode and submit the dot source or the exception."""
  try:
    queue.put(list(opt_util.DotControlFlowGraphsFromBytecode(bytecode)))
  except Exception as e:
    queue.put(DotControlFlowGraphsFromBytecodeError(bytecode, e))


def _ControlFlowGraphFromDotSourceToQueue(
  dot: str, queue: multiprocessing.Queue
) -> None:
  """Process a dot source and submit the CFG or the exception."""
  try:
    queue.put(ControlFlowGraphFromDotSource(dot))
  except Exception as e:
    queue.put(ControlFlowGraphFromDotSourceError(dot, e))


class ExceptionBuffer(Exception):
  """A meta-exception that is used to buffer multiple errors to be raised."""

  def __init__(self, errors):
    self.errors = errors


def ControlFlowGraphsFromBytecodes(
  bytecodes: typing.Iterator[str],
) -> typing.Iterator[cfg.ControlFlowGraph]:
  """A parallelised implementation of bytecode->CFG function."""
  dot_processes = []
  dot_queue = multiprocessing.Queue()
  for bytecode in bytecodes:
    process = multiprocessing.Process(
      target=_DotControlFlowGraphsFromBytecodeToQueue,
      args=(bytecode, dot_queue),
    )
    process.start()
    dot_processes.append(process)

  cfg_processes = []
  cfg_queue = multiprocessing.Queue()

  e = ExceptionBuffer([])

  for _ in range(len(dot_processes)):
    dots = dot_queue.get()
    if isinstance(dots, Exception):
      e.errors.append(dots)
    else:
      for dot in dots:
        process = multiprocessing.Process(
          target=_ControlFlowGraphFromDotSourceToQueue, args=(dot, cfg_queue)
        )
        process.start()
        cfg_processes.append(process)

  # Get the graphs generated by the CFG processes.
  for _ in range(len(cfg_processes)):
    graph = cfg_queue.get()
    if isinstance(graph, Exception):
      e.errors.append(graph)
    else:
      yield graph

  # Make sure that all processes have terminated. They will have, but best to
  # check.
  [p.join() for p in dot_processes]
  [p.join() for p in cfg_processes]

  if e.errors:
    raise e
