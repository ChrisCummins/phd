"""Utility code for working with LLVM."""

import collections
import multiprocessing
import pathlib
import re
import tempfile
import typing

import pydot
import pyparsing

from compilers.llvm import opt
from experimental.compilers.reachability import control_flow_graph as cfg
from labm8 import app
from labm8 import fs

FLAGS = app.FLAGS


def DotCfgsFromBytecode(bytecode: str) -> typing.Iterator[str]:
  """Create a control flow graph from an LLVM bytecode file.

  Args:
    bytecode: The LLVM bytecode to create CFG dots from.

  Returns:
    An iterator of dotfile strings.

  Raises:
    OptException: In case the opt pass fails.
    UnicodeDecodeError: If generated dotfile can't be read.
  """
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    output_dir = pathlib.Path(d)
    # Change into the output directory, because the -dot-cfg pass writes files
    # to the current working directory.
    with fs.chdir(output_dir):
      # We run with universal_newlines=False because the stdout of opt is the
      # binary bitcode, which we completely ignore (we're only interested in
      # stderr). This means we must encode stdin and decode stderr ourselves.
      process = opt.Exec(['-dot-cfg'],
                         stdin=bytecode.encode('utf-8'),
                         universal_newlines=False,
                         log=False)
      stderr = process.stderr.decode('utf-8')

      # Propagate failures from opt as OptExceptions.
      if process.returncode:
        raise opt.OptException(returncode=process.returncode, stderr=stderr)

      for file in output_dir.iterdir():
        # Opt pass prints the name of the dot files it generates, e.g.:
        #
        #     $ opt -dot-cfg < foo.ll
        #     WARNING: You're attempting to print out a bitcode file.
        #     This is inadvisable as it may cause display problems. If
        #     you REALLY want to taste LLVM bitcode first-hand, you
        #     can force output with the `-f' option.
        #
        #     Writing 'cfg.DoSomething.dot'...
        #     Writing 'cfg.main.dot'...
        if f"Writing '{file.name}'..." not in stderr:
          raise OSError(f"Could not find reference to file '{file.name}' in "
                        f"opt stderr:\n{process.stderr}")
        with open(file) as f:
          yield f.read()


def NodeAttributesToBasicBlock(
    node_attributes: typing.Dict[str, str]) -> typing.Dict[str, str]:
  """Get the basic block graph node attributes from a dot graph node.

  Args:
    node_attributes: The dictionary of dot graph information.

  Returns:
    A dictionary of node attributes.
  """
  label = node_attributes.get('label', '')
  if not label.startswith('"{'):
    raise ValueError(f"Unhandled label: '{label}'")
  # Lines are separated using '\l' in the dot label.
  lines = label.split('\l')
  return {
      # The name is in the first line.
      'name': lines[0][len('"{'):].split(':')[0],
      # All other lines except the last are either blank or contain instructions.
      'text': '\n'.join(x.lstrip() for x in lines[1:-1] if x.lstrip()),
  }


class LlvmControlFlowGraph(cfg.ControlFlowGraph):
  """A subclass of the generic control flow graph for LLVM CFGs.

  Each node in an LlvmControlFlowGraph has an additional "text" attribute which
  contains the LLVM instructions for the basic block as a string.
  """

  def BuildFullFlowGraph(self) -> 'LlvmControlFlowGraph':
    """Build a full program flow graph from the Control Flow Graph.

    This expands the control flow graph so that every node contains a single
    LLVM instruction. The "text" attribute of nodes contains the instruction.

    Unconditional branch instructions are ignored.

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
    sig = LlvmControlFlowGraph(name=self.graph['name'])

    # A global node count used to assign unique IDs to nodes in the new graph.
    sig_node_count = 0

    # When expanding the CFG to have a single block for every instruction, we
    # replace a single node with a contiguous run of nodes. We construct a map
    # of self.node IDs to (start,end) tuples of g.node IDs, allowing us to
    # translate node IDs for adding edges.
    NodeTranslationMapValue = collections.namedtuple('NodeTranslationMapValue',
                                                     ['start', 'end'])
    node_translation_map: typing.Dict[int, NodeTranslationMapValue] = {}

    # Iterate through all blocks in the source graph.
    for node, data in self.nodes(data=True):
      instructions = data['text'].split('\n')
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

        # Add a new node to the graph for the instruction.
        if (block_instruction_count == last_instruction and
            instruction.startswith('br ')):
          # Branches can either be conditional, e.g.
          #     br il %6, label %7, label %8
          # or unconditional, e.g.
          #     br label %9
          # Unconditional branches can be skipped - they contain no useful
          # information. Conditional branches can have the labels stripped.
          branch_instruction_components = instruction.split(', ')
          if len(branch_instruction_components) == 1:
            # Unconditional branches are ignored - they provide no meaningful
            # information beyond what the edge already includes.
            block_instruction_count -= 1
          else:
            # TODO(cec): Do we want to preserve the "true" "false" information
            # for outgoing edges? We currently throw it away.
            sig.add_node(
                new_node_id,
                name=new_node_name,
                text=branch_instruction_components[0])
        else:
          sig.add_node(new_node_id, name=new_node_name, text=instruction)

      # Add an entry to the node translation map for the start and end nodes
      # of this basic block.
      node_translation_map[node] = NodeTranslationMapValue(
          start=sig_node_count, end=sig_node_count + block_instruction_count)

      # Create edges between the sequential instruction nodes we just added
      # to the graph.
      [
          sig.add_edge(i, i + 1)
          for i in range(sig_node_count, sig_node_count +
                         block_instruction_count)
      ]

      # Update the global node count to be the value of the next unused node ID.
      sig_node_count += block_instruction_count + 1

    # Iterate through the edges in the source graph, translating their IDs to
    # IDs in the new graph using the node_translation_map.
    for src, dst in self.edges:
      new_src = node_translation_map[src].end
      new_dst = node_translation_map[dst].start
      sig.add_edge(new_src, new_dst)

    # Set the "entry" and "exit" blocks.
    new_entry_block = node_translation_map[self.entry_block].start
    sig.nodes[new_entry_block]['entry'] = True

    new_exit_block = node_translation_map[self.exit_block].end
    sig.nodes[new_exit_block]['exit'] = True

    return sig.ValidateControlFlowGraph(strict=False)

  def ValidateControlFlowGraph(self,
                               strict: bool = True) -> 'LlvmControlFlowGraph':
    """Validate the control flow graph."""
    super(LlvmControlFlowGraph, self).ValidateControlFlowGraph(strict=strict)

    # Check that each basic block has a text section.
    for _, data in self.nodes(data=True):
      if not data.get('text'):
        raise cfg.MalformedControlFlowGraphError(
            f"Missing 'text' attribute from node '{data}'")

    return self


def ControlFlowGraphFromDotSource(dot_source: str) -> LlvmControlFlowGraph:
  """Create a control flow graph from an LLVM-generated dot file.

  The control flow graph generated from the dot source is not guaranteed to
  be valid. That is, it may contain fusible basic blocks. This can happen if
  the creating the graph from unoptimized bytecode. To disable this generate
  the bytecode with optimizations enabled, e.g. clang -emit-llvm -O3 -S ...

  Args:
    dot_source: The dot source generated by the LLVM -dot-cfg pass.

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

  graph_re_match = re.match(r"\"CFG for '(\w+)' function\"", dot.get_name())
  if not graph_re_match:
    raise ValueError(f"Could not interpret graph name '{dot.get_name()}'")

  # Create the ControlFlowGraph instance.
  graph = LlvmControlFlowGraph(name=graph_re_match.group(1))

  # Create the nodes and build a map from node names to indices.
  node_name_to_index_map = {}
  for i, node in enumerate(dot.get_nodes()):
    if node.get_name() in node_name_to_index_map:
      raise ValueError(f"Duplicate node name! '{node.get_name()}'")
    node_name_to_index_map[node.get_name()] = i
    graph.add_node(i, **NodeAttributesToBasicBlock(node.get_attributes()))

  def NodeIndex(node: pydot.Node) -> int:
    """Get the index of a node."""
    return node_name_to_index_map[node.get_name()]

  first_node_name = sorted(node_name_to_index_map.keys())[0]
  entry_block = dot.get_node(first_node_name)[0]
  graph.nodes[NodeIndex(entry_block)]['entry'] = True

  def IsExitNode(node: pydot.Node) -> bool:
    """Determine if the given node is an exit block.

    In LLVM bytecode, an exit block is one in which the final instruction begins
    with 'ret '. There should be only one exit block per graph.
    """
    label = node.get_attributes().get('label', '')
    # Node labels use \l to escape newlines.
    label_lines = label.split('\l')
    # The very last line is just a closing brace.
    last_line_with_instructions = label_lines[-2]
    return last_line_with_instructions.lstrip().startswith('ret ')

  # Set the exit node.
  exit_nodes = []
  for node in dot.get_nodes():
    if IsExitNode(node):
      exit_nodes.append(node)

  if len(exit_nodes) != 1:
    raise ValueError("Couldn't find an exit block")

  graph.nodes[NodeIndex(exit_nodes[0])]['exit'] = True

  for edge in dot.get_edges():
    # In the dot file, an edge looks like this:
    #     Node0x7f86c670c590:s0 -> Node0x7f86c65001a0;
    # We split the :sX suffix from the source to get the node name.
    # TODO(cec): We're currently throwing away the subrecord information and
    # True/False labels on edges. We may want to preserve that here.
    src = node_name_to_index_map[edge.get_source().split(':')[0]]
    dst = node_name_to_index_map[edge.get_destination()]
    graph.add_edge(src, dst)

  return graph


class DotCfgsFromBytecodeError(ValueError):
  """An error raised processing a bytecode file."""

  def __init__(self, bytecode: str, error: Exception):
    self.input = bytecode
    self.error = error


class ControlFlowGraphFromDotSourceError(ValueError):
  """An error raised processing a dot source."""

  def __init__(self, dot: str, error: Exception):
    self.input = dot
    self.error = error


def _DotCfgsFromBytecodeToQueue(bytecode: str,
                                queue: multiprocessing.Queue) -> None:
  """Process a bytecode and submit the dot source or the exception."""
  try:
    queue.put(list(DotCfgsFromBytecode(bytecode)))
  except Exception as e:
    queue.put(DotCfgsFromBytecodeError(bytecode, e))


def _ControlFlowGraphFromDotSourceToQueue(dot: str,
                                          queue: multiprocessing.Queue) -> None:
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
    bytecodes: typing.Iterator[str]) -> typing.Iterator[cfg.ControlFlowGraph]:
  dot_processes = []
  dot_queue = multiprocessing.Queue()
  for bytecode in bytecodes:
    process = multiprocessing.Process(
        target=_DotCfgsFromBytecodeToQueue, args=(bytecode, dot_queue))
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
            target=_ControlFlowGraphFromDotSourceToQueue, args=(dot, cfg_queue))
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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.RunWithArgs(main)
