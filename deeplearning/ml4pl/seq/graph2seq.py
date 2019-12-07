"""Module for conversion from labelled graphs to encoded sequences."""
import collections
import enum
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import lru
import numpy as np

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app

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
  # nodes in teh graph.
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
    cache_size: Optional[int] = None,
  ):
    self.graph_db = graph_db
    self.cache_size = cache_size or FLAGS.graph2seq_cache_entries

  def Encode(
    self, ids: List[int]
  ) -> Union[List[np.array], EncodedSubsequences]:
    """Translate a list of graph IDs to encoded sequences."""
    raise NotImplementedError("abstract class")


class GraphToEncodedSequence(EncoderBase):
  """Encode a graph to a single encoded sequence.

  Uses the original intermediate representation to produce the tokenized
  sequence, entirely discarding the graph structure.
  """

  def __init__(
    self,
    graph_db: graph_tuple_database.Database,
    ir2seq_encoder: ir2seq.EncoderBase,
    cache_size: Optional[int] = None,
  ):
    super(GraphToEncodedSequence, self).__init__(
      graph_db, cache_size=cache_size
    )

    self.ir2seq_encoder = ir2seq_encoder

    # Maintain a mapping from graph IDs to IR IDs.
    self.graph_id_to_ir_id: Dict[int, int] = lru.LRU(
      FLAGS.graph2seq_cache_entries
    )

    # Maintain a mapping from IR IDs to encoded sequences to amortize the
    # cost of encoding.
    self.ir_id_to_encoded: Dict[int, np.array] = lru.LRU(
      FLAGS.graph2seq_cache_entries
    )

  def Encode(self, ids: List[int]) -> List[np.array]:
    """Return encoded sequences for the given graph IDs.

    This adapts the methodology used in the PACT'17 "DeepTune" paper to LLVM
    IR. It provides a tokenized list of vocabulary indices from bytecodes, which
    can then be processed by sequential models.

    Args:
      ids: A list of graphs to encode.

    Returns:
      A list of encoded sequences.
    """
    unique_graph_ids = set(ids)

    unknown_graph_ids = [
      graph_id
      for graph_id in unique_graph_ids
      if graph_id not in self.graph_id_to_ir_id
    ]

    if unknown_graph_ids:
      # Look the IR IDs of any unknown graphs.
      with self.graph_db.Session() as session:
        graph_id_to_ir_id = {
          row.id: row.ir_id
          for row in session.query(
            graph_tuple_database.GraphTuple.id,
            graph_tuple_database.GraphTuple.ir_id,
          ).filter(
            graph_tuple_database.GraphTuple.id.in_(unknown_graph_ids)
          )
        }
      if len(graph_id_to_ir_id) != len(unknown_graph_ids):
        raise KeyError(
          f"Requested {len(unknown_graph_ids)} graph IDs but received "
          f"{len(graph_id_to_ir_id)}"
        )
      # Add these unknown IDs to teh cache.
      for graph_id, ir_id in graph_id_to_ir_id.items():
        self.graph_id_to_ir_id[graph_id] = ir_id

      # Create reverse mapping from IR ID to a list of graph IDs because one
      # bytecode can map to multiple graphs.
      ir_id_to_graph_id: Dict[int, List[int]] = collections.defaultdict(list)
      for graph_id, ir_id in graph_id_to_ir_id.items():
        ir_id_to_graph_id[ir_id].append(graph_id)

      # Encode the unknown IRs.
      sorted_ir_ids_to_encode = sorted(
        [
          ir_id
          for ir_id in ir_id_to_graph_id.keys()
          if ir_id not in self.ir_id_to_encoded
        ]
      )
      sorted_encoded_sequences = self.ir2seq_encoder.Encode(
        sorted_ir_ids_to_encode
      )

      # Cache the encoded unknown IRs.
      for ir_id, encoded_sequence in zip(
        sorted_ir_ids_to_encode, sorted_encoded_sequences
      ):
        self.ir_id_to_encoded[ir_id] = encoded_sequence

    return [
      self.ir_id_to_encoded[self.graph_id_to_ir_id[graph_id]]
      for graph_id in ids
    ]


# TODO(github.com/ChrisCummins/ProGraML/issues/24): Implement in order to
# support node-level LSTM models:
# class GraphToEncodedSubsequences(EncoderBase):
#   """Encode graphs to per-node sub-sequences.
#
#   This uses the graph structure to produce a tokenized sequence ordered by
#   depth first traversal, and allows mapping from graph nodes to sub-seqeuences
#   within the encoded output.
#   """
#
#   def __init__(
#     self,
#     graph_db: graph_tuple_database.Database,
#     ir2seq_encoder: ir2seq.EncoderBase,
#     subsequences_type: SubsequencesType,
#     cache_size: Optional[int] = None,
#   ):
#     super(GraphToEncodedSubsequences, self).__init__(graph_db,
#                                                    cache_size=cache_size)
#
#     self.ir2seq_encoder = ir2seq_encoder
#
#     if subsequences_type == SubsequencesType.STATEMENT:
#       self.graph2seq_encoder = self._GraphToEncodedStatementGroups
#     elif subsequences_type == SubsequencesType.IDENTIFIER:
#       self.graph2seq_encoder = self._GraphToEncodedIdentifierGroups
#     else:
#       raise NotImplementedError("unreachable")
#
#     # Maintain a mapping from graph IDs to IR IDs.
#     self.graph_id_to_ir_id: Dict[int, int] = lru.LRU(
#         FLAGS.graph2seq_cache_entries)
#
#   def Encode(
#     self, graph_ids: List[int]
#   ) -> Tuple[
#     Dict[int, np.array],
#     Dict[int, np.array],
#     Dict[int, np.array],
#   ]:
#     """Serialize a graph into an encoded sequence.
#
#     This method is used to provide a serialized sequence of encoded tokens
#     of the statements in a program graph that can be processed sequentially,
#     and a grouping of encoded tokens to statements.
#
#     For example, the graph of this function:
#
#       define i32 @B() #0 {
#         %1 = alloca i32, align 4
#         store i32 0, i32* %1, align 4
#         ret i32 15
#       }
#
#     would comprise three statement nodes, and additional data nodes for the
#     statements' operands (e.g. %1). A pre-order depth first traversal of the
#     graph produces the linear ordering of statement nodes, which are then
#     encoded and concatenated to produce a list of vocabulary indices such as:
#
#       [
#          0, 1, 2, 1, 4, 5, 6,  # encoded `%1 = alloca i32, align 4`
#          12, 9, 3,             # encoded `store i32 0, i32* %1, align 4`
#          9, 8,                 # encoded `ret i32 15`
#       ]
#
#     This encoded sequence can then be grouped into the three individual
#     statements by assigning each statement a unique ID, such as:
#
#       [
#         0, 0, 0, 0, 0, 0, 0,
#         1, 1, 1,
#         2, 2,
#       ]
#
#     This method computes and returns these two arrays, along with a third array
#     which contains a masking of nodes from the input program graph, marking the
#     non-statement nodes as inactive. E.g. for a graph with 5 statement nodes
#     and 3 data nodes, the mask will consist of 8 boolean values, 5 True, 3
#     False. Use this array to mask a 'node_y' label list to exclude the labels
#     for non-statement nodes.
#
#     Args:
#       graph_ids: A list of graph IDs.
#       group_by: The method used to group statements. There are two options:
#         "statement", in which each statement is it's own group, and
#         "identifier", in which all statements which reference an identifier are
#         grouped.
#
#     Returns:
#       A tuple of <encoded_sequences, statement_groupings, node_mask>
#       dictionaries, which map graph_id to encoded sequences and statement
#       groupings (2D matrices of shape [len(graph_ids),
#       self.max_sequence_length]), and node_mask arrays of shape
#       [len(graph_ids),?] which list the nodes which are selected from
#       each graph.
#     """
#     # Update the mapping from graph to bytecode IDs.
#     unknown_graph_ids = [
#       id_ for id_ in graph_ids if id_ not in self.graph_to_ir_ids
#     ]
#
#     # Lookup the bytecode IDs.
#     with self.graph_db.Session() as session:
#       query = session.query(
#         graph_tuple_database.GraphTuple.id, graph_tuple_database.GraphTuple.ir_id
#       ).filter(graph_tuple_database.GraphTuple.id.in_(unknown_graph_ids))
#       for graph_id, ir_id in query:
#         self.graph_to_ir_ids[graph_id] = ir_id
#
#     ir_ids = [
#       self.graph_to_ir_ids[graph_id] for graph_id in graph_ids
#     ]
#
#     # Fetch the requested unlabelled graphs.
#     graph_ids_to_fetch = set(ir_ids)
#
#     # Fetch the graph data.
#     with self.unlabelled_graph_db.Session() as session:
#       # TODO(github.com/ChrisCummins/ProGraML/issues/20): Rebuild networkx from
#       # graph tuples.
#       query = (
#         session.query(
#           graph_tuple_database.GraphTuple.ir_id,
#           graph_database.Graph.pickled_data,
#         )
#         .join(graph_database.Graph)
#         .filter(graph_tuple_database.GraphTuple.ir_id.in_(graph_ids_to_fetch))
#       )
#
#       ids_to_graphs = {
#         ir_id: pickle.loads(data) for ir_id, data in query
#       }
#
#     if len(graph_ids_to_fetch) != len(ids_to_graphs):
#       raise EnvironmentError(
#         f"Graph IDs not found in database {self.graph_db.url}: "
#         f"{set(graph_ids_to_fetch) - set(ids_to_graphs.keys())}"
#       )
#
#     # Encode the graphs
#     ids_to_encoded_sequences = {}
#     ids_to_grouping_ids = {}
#     ids_to_node_masks = {}
#     for ir_id, graph in ids_to_graphs.items():
#       seqs, ids, node_mask = self.graph2seq_encoder(graph)
#       ids_to_encoded_sequences[ir_id] = seqs
#       ids_to_grouping_ids[ir_id] = ids
#       ids_to_node_masks[ir_id] = node_mask
#
#     return ids_to_encoded_sequences, ids_to_grouping_ids, ids_to_node_masks
#
#   def EncodeBytecodes(self, ir_ids: List[int]):
#     with self.unlabelled_graph_db.Session() as session:
#       # TODO(github.com/ChrisCummins/ProGraML/issues/20): Rebuild networkx from
#       # graph tuples.
#       query = (
#         session.query(
#           graph_tuple_database.GraphTuple.ir_id,
#           graph_database.Graph.pickled_data,
#         )
#         .join(graph_database.Graph)
#         .filter(graph_tuple_database.GraphTuple.ir_id.in_(ir_ids))
#       )
#
#       ids_to_graphs = {
#         ir_id: pickle.loads(data) for ir_id, data in query
#       }
#
#     if len(ir_ids) != len(ids_to_graphs):
#       raise EnvironmentError(
#         f"Graph IDs not found in database {self.graph_db.url}: "
#         f"{set(graph_ids_to_fetch) - set(ids_to_graphs.keys())}"
#       )
#
#     encoded_sequences, segment_ids, node_masks = [], [], []
#
#     for ir_id in ir_ids:
#       graph = ids_to_graphs[ir_id]
#       seqs, ids, node_mask = self.graph2seq_encoder(graph)
#       encoded_sequences.append(seqs)
#       segment_ids.append(ids)
#       node_masks.append(node_mask)
#
#     return encoded_sequences, segment_ids, node_masks
#
#   @property
#   def unlabelled_graph_db(self) -> graph_tuple_database.Database:
#     """Get the database of unlabelled graphs."""
#     if self._unlabelled_graph_db:
#       return self._unlabelled_graph_db
#     elif FLAGS.unlabelled_graph_db:
#       self._unlabelled_graph_db = FLAGS.unlabelled_graph_db()
#       return self._unlabelled_graph_db
#     else:
#       raise app.UsageError("--unlabelled_graph_db must be set")
#
#   def _EncodeStringsWithGroupings(
#     self, strings: List[str]
#   ) -> Tuple[np.array, np.array]:
#     """Encode the given strings and return a flattened list of the encoded
#     values, along with grouping IDs.
#     """
#     encoded_sequences = self.ir2seq_encoder.EncodeBytecodeStrings(
#       strings, pad=False
#     )
#     statement_indices = []
#     for i, enc in enumerate(encoded_sequences):
#       statement_indices.append([i] * len(enc))
#
#     if not encoded_sequences == []:
#       encoded_sequences = np.concatenate(encoded_sequences)
#     if not statement_indices == []:
#       statement_indices = np.concatenate(statement_indices)
#
#     return encoded_sequences, np.array(statement_indices, dtype=np.int32)
#
#   def _GraphToEncodedStatementGroups(
#     self, graph: nx.MultiDiGraph
#   ) -> Tuple[np.array, np.array, np.array]:
#     """Serialize the graph to an encoded sequence and set of statement indices.
#     """
#     serialized_node_list = list(cdfg.SerializeToStatementList(graph))
#     statement_nodes = set(serialized_node_list)
#     node_mask = np.array(
#       [1 if node in statement_nodes else 0 for node in graph.nodes()],
#       dtype=np.int32,
#     )
#
#     if not any(node_mask):
#       return (
#         np.array([], dtype=np.int32),
#         np.array([], dtype=np.int32),
#         np.array([], dtype=np.int32),
#       )
#
#     strings_to_encode = [
#       graph.nodes[n].get("original_text", "") for n in serialized_node_list
#     ]
#
#     seqs, ids = self._EncodeStringsWithGroupings(strings_to_encode)
#
#     if max(ids) > graph.number_of_nodes():
#       app.Error(
#         "Found max ID %s in graph of only %s nodes",
#         max(ids),
#         graph.number_of_nodes(),
#       )
#
#     return seqs, ids, node_mask
#
#   def _GraphToEncodedIdentifierGroups(
#     self, graph: nx.MultiDiGraph
#   ) -> Tuple[np.array, np.array, np.array]:
#     """Serialize the graph to an encoded sequence and set of statement indices.
#     """
#     identifiers, node_mask = [], []
#     for node, type_ in graph.nodes(data="type"):
#       if type_ == "identifier":
#         identifiers.append(node)
#         node_mask.append(1)
#       else:
#         node_mask.append(0)
#     node_mask = np.array(node_mask, dtype=np.int32)
#
#     if not any(node_mask):
#       return (
#         np.array([], dtype=np.int32),
#         np.array([], dtype=np.int32),
#         np.array([], dtype=np.int32),
#       )
#
#     strings_to_encode = labtypes.flatten(
#       [
#         [
#           graph.nodes[n].get("original_text", "")
#           for n in graph_query.GetStatementsForNode(graph, identifier)
#         ]
#         for identifier in identifiers
#       ]
#     )
#
#     seqs, ids = self._EncodeStringsWithGroupings(strings_to_encode)
#
#     if max(ids) > graph.number_of_nodes():
#       app.Error(
#         "Found max ID %s in graph of only %s nodes",
#         max(ids),
#         graph.number_of_nodes(),
#       )
#
#     return seqs, ids, node_mask
