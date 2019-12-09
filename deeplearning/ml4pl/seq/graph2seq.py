"""Module for conversion from labelled graphs to encoded sequences."""
import enum
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import lru
import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import progress

FLAGS = app.FLAGS

app.DEFINE_integer(
  "graph2seq_cache_entries",
  512000,
  "The number of ID -> encoded sequence entries to cache.",
)


class SubsequencesType(enum.Enum):
  """Determine the type of subsequences to generate for a graph."""

  # Produce one subsequence per statement in the graph.
  STATEMENT = 1
  # Produce a subsequence of all statements for a particular identifier.
  IDENTIFIER = 2


class EncodedSubsequences(NamedTuple):
  """An encoded sequence representing a graph, segmented into sub-sequences
  representing either individual statements or identifiers.
  """

  # An array of encoded sequences representing an encoded serialization of the
  # graph.
  # Shape (?, vocab_size), dtype int32
  encoded_sequence: np.array
  # An array of segment IDs in the range [0, node_count] which groups the
  # encoded_sequence array into one sub-sequence for every node in the graph.
  segment_ids: np.array
  # An array of shape (node_count) of [0, 1] values indicating whether a
  # particular node in the graph should be used as label. For statement-level
  # groupings, statement_count values in this mask will be set. For
  # identifier-level groupings, identifier_count values in this mask will be
  # set.
  node_mask: np.array


class EncoderBase(object):
  """Base class for performing graph-to-encoded sequence translation."""

  def __init__(
    self,
    graph_db: graph_tuple_database.Database,
    ir2seq_encoder: ir2seq.EncoderBase,
    cache_size: Optional[int] = None,
  ):
    self.graph_db = graph_db
    self.ir2seq_encoder = ir2seq_encoder

    # Maintain a mapping from IR IDs to encoded sequences to amortize the
    # cost of encoding.
    cache_size = cache_size or FLAGS.graph2seq_cache_entries
    self.ir_id_to_encoded: Dict[int, np.array] = lru.LRU(cache_size)

  @property
  def max_encoded_length(self) -> int:
    """Return an upper bound on the length of the encoded sequences."""
    return self.ir2seq_encoder.max_encoded_length

  @property
  def vocabulary_size(self) -> int:
    """Get the size of the vocabulary, including the unknown-vocab element."""
    return self.ir2seq_encoder.vocabulary_size

  def Encode(
    self,
    graphs: List[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[Union[np.array, EncodedSubsequences]]:
    """Translate a list of graph IDs to encoded sequences."""
    raise NotImplementedError("abstract class")


class GraphEncoder(EncoderBase):
  """Encode a graph to a single encoded sequence.

  Uses the original intermediate representation to produce the tokenized
  sequence, entirely discarding the graph structure.
  """

  def Encode(
    self,
    graphs: List[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[np.array]:
    """Return encoded sequences for the given graph IDs.

    This adapts the methodology used in the PACT'17 "DeepTune" paper to LLVM
    IR. It provides a tokenized list of vocabulary indices from bytecodes, which
    can then be processed by sequential models.

    Args:
      ids: A list of graphs to encode.

    Returns:
      A list of encoded sequences.
    """
    unknown_ir_ids = set(
      [
        graph.ir_id
        for graph in graphs
        if graph.ir_id not in self.ir_id_to_encoded
      ]
    )

    if unknown_ir_ids:
      # Encode the unknown IRs.
      sorted_ir_ids_to_encode = sorted(unknown_ir_ids)
      sorted_encoded_sequences = self.ir2seq_encoder.Encode(
        sorted_ir_ids_to_encode, ctx=ctx
      )

      # Cache the encoded unknown IRs.
      for ir_id, encoded_sequence in zip(
        sorted_ir_ids_to_encode, sorted_encoded_sequences
      ):
        self.ir_id_to_encoded[ir_id] = encoded_sequence

    return [self.ir_id_to_encoded[graph.ir_id] for graph in graphs]


class StatementEncoderBase(EncoderBase):
  """Encode graphs to per-node sub-sequences.

  This uses the graph structure to produce a tokenized sequence ordered by
  depth first traversal, and allows mapping from graph nodes to sub-seqeuences
  within the encoded output.
  """

  def GraphToSubsequences(
    self,
    graph: nx.MultiDiGraph,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> EncodedSubsequences:
    raise NotImplementedError("abstract class")

  def Encode(
    self,
    graphs: List[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[EncodedSubsequences]:
    """Serialize a graph into an encoded sequence.

    This method is used to provide a serialized sequence of encoded tokens
    of the statements in a program graph that can be processed sequentially,
    and a grouping of encoded tokens to statements.

    For example, the graph of this function:

      define i32 @B() #0 {
        %1 = alloca i32, align 4
        store i32 0, i32* %1, align 4
        ret i32 15
      }

    would comprise three statement nodes, and additional data nodes for the
    statements' operands (e.g. %1). A pre-order depth first traversal of the
    graph produces the linear ordering of statement nodes, which are then
    encoded and concatenated to produce a list of vocabulary indices such as:

      [
         0, 1, 2, 1, 4, 5, 6,  # encoded `%1 = alloca i32, align 4`
         12, 9, 3,             # encoded `store i32 0, i32* %1, align 4`
         9, 8,                 # encoded `ret i32 15`
      ]

    This encoded sequence can then be grouped into the three individual
    statements by assigning each statement a unique ID, such as:

      [
        0, 0, 0, 0, 0, 0, 0,
        1, 1, 1,
        2, 2,
      ]

    This method computes and returns these two arrays, along with a third array
    which contains a masking of nodes from the input program graph, marking the
    non-statement nodes as inactive. E.g. for a graph with 5 statement nodes
    and 3 data nodes, the mask will consist of 8 boolean values, 5 True, 3
    False. Use this array to mask a 'node_y' label list to exclude the labels
    for non-statement nodes.

    Args:
      graphs: A list of graphs to encode.

    Returns:
      A list of EncodedSubsequence tuples, where each tuple maps a graph to
      encoded sequences, subsequence groupings, and node_mask arrays which list
      the nodes which are selected from each graph.
    """
    # Look for any graphs that are unknown.
    graphs_to_encode = {
      graph.ir_id: graph
      for graph in graphs
      if graph.ir_id not in self.ir_id_to_encoded
    }

    if graphs_to_encode:
      # Encode and cache the unknown graphs.
      for ir_id, graph in graphs_to_encode.items():
        self.ir_id_to_encoded[ir_id] = self.GraphToSubsequences(
          graph.tuple.ToNetworkx(), ctx=ctx
        )

    return [self.ir_id_to_encoded[graph.ir_id] for graph in graphs]

  def EncodeWithSegmentIds(
    self,
    strings: List[str],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> Tuple[np.array, np.array]:
    """Encode the given strings and return a flattened list of the encoded
    values and the segment IDs.
    """
    encoded_sequences = self.ir2seq_encoder.Encode(strings, ctx=ctx)
    statement_indices = []
    for i, enc in enumerate(encoded_sequences):
      statement_indices.append([i] * len(enc))

    if encoded_sequences:
      encoded_sequences = np.concatenate(encoded_sequences)
    if statement_indices:
      statement_indices = np.concatenate(statement_indices)

    return encoded_sequences, statement_indices


class StatementEncoder(StatementEncoderBase):
  def GraphToSubsequences(
    self,
    graph: nx.MultiDiGraph,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> EncodedSubsequences:
    """Serialize a graph to an encoded sequence of statements."""
    serialized_node_list = list(SerializeGraphToStatementNodes(graph))

    # Create a mask of the graph nodes which are statements.
    statement_nodes = set(serialized_node_list)
    statement_node_mask = np.array(
      [1 if node in statement_nodes else 0 for node in graph.nodes()],
      dtype=np.int32,
    )

    # The graph has no statements!
    if not any(statement_node_mask):
      raise ValueError("Graph contains no statement nodes")

    strings_to_encode = [
      graph.nodes[node]["text"] for node in serialized_node_list
    ]

    encoded_sequence, segment_ids = self.EncodeWithSegmentIds(
      strings_to_encode, ctx=ctx
    )

    if segment_ids[-1] > graph.number_of_nodes():
      raise ValueError(
        f"Found max segment ID {segment_ids[-1]} in graph with "
        f"only {graph.number_of_nodes()} nodes"
      )

    return EncodedSubsequences(
      encoded_sequence=encoded_sequence,
      segment_ids=segment_ids,
      node_mask=statement_node_mask,
    )


class IdentifierEncoder(StatementEncoderBase):
  def GraphToSubsequences(
    self,
    graph: nx.MultiDiGraph,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> EncodedSubsequences:
    """Serialize a graph to an encoded sequence of identifier statement groups.
    """
    identifier_nodes, identifier_node_mask = [], []
    for node, type in graph.nodes(data="type"):
      if type == programl_pb2.Node.IDENTIFIER:
        identifier_nodes.append(node)
        identifier_node_mask.append(1)
      else:
        identifier_node_mask.append(0)
    identifier_node_mask = np.array(identifier_node_mask, dtype=np.int32)

    if not any(identifier_node_mask):
      raise ValueError("Graph contains no identifier nodes")

    strings_to_encode = []
    for identifier in identifier_nodes:
      strings_to_encode += [
        graph.nodes[n]["text"] for n in GetStatementsForNode(graph, identifier)
      ]

    encoded_sequence, segment_ids = self.EncodeWithSegmentIds(
      strings_to_encode, ctx=ctx
    )

    if segment_ids[-1] > graph.number_of_nodes():
      raise ValueError(
        f"Found max segment ID {segment_ids[-1]} in graph with "
        f"only {graph.number_of_nodes()} nodes"
      )

    return EncodedSubsequences(
      encoded_sequence=encoded_sequence,
      segment_ids=segment_ids,
      node_mask=identifier_node_mask,
    )


def SerializeGraphToStatementNodes(g: nx.MultiDiGraph) -> Iterable[int]:
  """Walk a program graph and emit the node IDs of statements using depth-first
  traversal order.

  Returns:
    An iterator over node IDs.
  """
  visited_statements: Set[int] = set()
  visited_functions: Set[str] = set()

  # Maintain a stack of functions entry nodes to visit, starting at the root.
  function_entry_stack = [0]

  while function_entry_stack:
    function_entry_node = function_entry_stack[-1]
    function_entry_stack.pop()

    visited_functions.add(function_entry_node)

    # Maintain a stack of statement nodes to visit, starting at the function
    # entry.
    statement_node_stack = [function_entry_node]

    # Pre-order depth first graph traversal to emit the strings.
    while statement_node_stack:
      node = statement_node_stack[-1]
      statement_node_stack.pop()

      visited_statements.add(node)
      yield node
      for _, dst, flow in g.out_edges(node, data="flow"):
        if flow == programl_pb2.Edge.CONTROL:
          # Follow control edges.
          if dst not in visited_statements:
            statement_node_stack.append(dst)
        elif flow == programl_pb2.Edge.CALL:
          # Make a not of call edges to be visited later.
          if dst not in visited_functions:
            function_entry_stack.append(dst)


def GetStatementsForNode(graph: nx.MultiDiGraph, node: int) -> Iterable[int]:
  """Return the statements which are connected to the given node.

  Args:
    graph: The graph to fetch the statements from.
    node: The node to fetch the statements for. If the node is a statement, it
      returns itself. If the node is an identifier, it returns all statements
      which define/use this identifier.

  Returns:
    An iterator over statement nodes.
  """
  root = graph.nodes[node]
  if root["type"] == programl_pb2.Node.STATEMENT:
    yield node
  elif root["type"] == programl_pb2.Node.IDENTIFIER:
    for src, _ in graph.in_edges(node):
      if graph.nodes[src]["type"] == programl_pb2.Node.STATEMENT:
        yield src
    for _, dst in graph.out_edges(node):
      if graph.nodes[dst]["type"] == programl_pb2.Node.STATEMENT:
        yield dst
  else:
    raise ValueError(f"Unknown statement type `{root['type']}`")
