"""Module for conversion from labelled graphs to encoded sequences."""
import collections
import pickle
import typing

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_query
from deeplearning.ml4pl.graphs.unlabelled.cdfg import (
  control_and_data_flow_graph as cdfg,
)
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import labtypes

FLAGS = app.FLAGS

app.DEFINE_database(
  "unlabelled_graph_db",
  graph_database.Database,
  None,
  "URL of unlabelled graphs to read bytecodes from.",
  must_exist=True,
)


class EncoderBase(object):
  """Base class for performing graph-to-encoded sequence translation.

  This graph exposes two methods for converting graphs to encoded sequences:

    1. GraphsToEncodedBytecodes() which uses the original bytecode source to
        produce the tokenized sequence, entirely discarding the graph, and
    2. GraphsToEncodedStatementGroups() which uses the graph structure to
        produce a tokenized sequence ordered by depth first traversal, and
        allowing the mapping from graph nodes to sub-seqeuences within the
        encoded output.

  In both cases, the data layout used in this project is inefficiently designed
  and requires SQL queries across multiple databases. There is a signifcant
  overhead to using this class.
  """

  def __init__(self, graph_db: graph_database.Database):
    self.graph_db = graph_db

  def Encode(self, graph_ids: typing.List[int]) -> typing.Any:
    raise NotImplementedError("abstract class")


class GraphToBytecodeEncoder(EncoderBase):
  """Encode graphs to bytecode sequences.

  """

  def __init__(
    self,
    graph_db: graph_database.Database,
    bytecode_encoder: ir2seq.EncoderBase,
  ):
    super(GraphToBytecodeEncoder, self).__init__(graph_db)

    self.bytecode_encoder = bytecode_encoder

    # Maintain a mapping from graph IDs to encoded bytecodes to amortize the
    # cost of encoding.
    self.graph_to_encoded_bytecode: typing.Dict[int, np.array] = {}

  def Encode(self, graph_ids: typing.List[int]):
    """Return encoded bytecodes for the given graph IDs.

    This adapts the methodology used in the PACT'17 "DeepTune" paper to LLVM
    IR. It provides a tokenized list of vocabulary indices from bytecodes, which
    can then be processed by sequential models.

    Args:
      graph_ids: A list of graphs to fetch the encoded bytecodes for.

    Returns:
      A list of encoded bytecodes.
    """
    unknown_graph_ids = [
      graph_id
      for graph_id in graph_ids
      if graph_id not in self.graph_to_encoded_bytecode
    ]

    if unknown_graph_ids:
      # we hope this will not be printed at epoch 2...
      app.Log(
        1,
        "unknown_graph_ids has length %s and ids %s",
        len(unknown_graph_ids),
        unknown_graph_ids,
      )

      # Look the bytecode IDs of any unknown graphs.
      with self.graph_db.Session() as session:
        query = session.query(
          graph_database.GraphMeta.id, graph_database.GraphMeta.bytecode_id
        )
        query = query.filter(graph_database.GraphMeta.id.in_(unknown_graph_ids))
        graph_to_bytecode_id = {
          graph_id: bytecode_id for graph_id, bytecode_id in query
        }
      if len(graph_to_bytecode_id) != len(unknown_graph_ids):
        raise EnvironmentError(
          f"len(graph_to_bytecode_id)={len(graph_to_bytecode_id)} != "
          f"len(unknown_graph_ids)={len(unknown_graph_ids)}"
        )

      # Create reverse mapping from bytecode ID to a list of graph IDs. One
      # bytecode can map to multiple graphs.
      bytecode_to_graph_ids: typing.Dict[
        int, typing.List[int]
      ] = collections.defaultdict(list)
      for graph_id, bytecode_id in graph_to_bytecode_id.items():
        bytecode_to_graph_ids[bytecode_id].append(graph_id)

      bytecodes_to_encode = sorted(list(bytecode_to_graph_ids.keys()))
      encoded_sequences = self.bytecode_encoder.Encode(bytecodes_to_encode)

      for bytecode_id, encoded_sequence in zip(
        bytecodes_to_encode, encoded_sequences
      ):
        graph_ids_for_bytecode = bytecode_to_graph_ids[bytecode_id]
        for graph_id in graph_ids_for_bytecode:
          self.graph_to_encoded_bytecode[graph_id] = encoded_sequence

    return [self.graph_to_encoded_bytecode[graph_id] for graph_id in graph_ids]


class EncodedBytecodeGrouping(typing.NamedTuple):
  """An encoded bytecode with node segments."""

  encoded_sequences: np.array
  segment_ids: np.array
  node_mask: np.array


class GraphToBytecodeGroupingsEncoder(EncoderBase):
  """Encode graphs to bytecode sequences with statement groupings."""

  def __init__(
    self,
    graph_db: graph_database.Database,
    bytecode_encoder: ir2seq.EncoderBase,
    group_by: str,
  ):
    super(GraphToBytecodeGroupingsEncoder, self).__init__(graph_db)

    self._unlabelled_graph_db = None

    self.bytecode_encoder = bytecode_encoder

    if group_by == "statement":
      self.encode_graph = self._GraphToEncodedStatementGroups
    elif group_by == "identifier":
      self.encode_graph = self._GraphToEncodedIdentifierGroups
    else:
      raise ValueError(
        "Unknown option for `group_by`. Expected one of "
        "{statement,identifier}"
      )

    self.graph_to_bytecode_ids = {}

    # TODO(github.com/ChrisCummins/ProGraML/issues/20): Implement LRU cache for
    # encoded bytecodes.
    self.graph_to_bytecode_grouping: typing.Dict[
      int, EncodedBytecodeGrouping
    ] = {}

  def Encode(
    self, graph_ids: typing.List[int]
  ) -> typing.Tuple[
    typing.Dict[int, np.array],
    typing.Dict[int, np.array],
    typing.Dict[int, np.array],
  ]:
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
      graph_ids: A list of graph IDs.
      group_by: The method used to group statements. There are two options:
        "statement", in which each statement is it's own group, and
        "identifier", in which all statements which reference an identifier are
        grouped.

    Returns:
      A tuple of <encoded_sequences, statement_groupings, node_mask>
      dictionaries, which map graph_id to encoded sequences and statement
      groupings (2D matrices of shape [len(graph_ids),
      self.max_sequence_length]), and node_mask arrays of shape
      [len(graph_ids),?] which list the nodes which are selected from
      each graph.
    """
    # Update the mapping from graph to bytecode IDs.
    unknown_graph_ids = [
      id_ for id_ in graph_ids if id_ not in self.graph_to_bytecode_ids
    ]

    # Lookup the bytecode IDs.
    with self.graph_db.Session() as session:
      query = session.query(
        graph_database.GraphMeta.id, graph_database.GraphMeta.bytecode_id
      ).filter(graph_database.GraphMeta.id.in_(unknown_graph_ids))
      for graph_id, bytecode_id in query:
        self.graph_to_bytecode_ids[graph_id] = bytecode_id

    bytecode_ids = [
      self.graph_to_bytecode_ids[graph_id] for graph_id in graph_ids
    ]

    # Fetch the requested unlabelled graphs.
    graph_ids_to_fetch = set(bytecode_ids)

    # Fetch the graph data.
    with self.unlabelled_graph_db.Session() as session:
      # TODO(github.com/ChrisCummins/ProGraML/issues/20): Rebuild networkx from
      # graph tuples.
      query = (
        session.query(
          graph_database.GraphMeta.bytecode_id,
          graph_database.Graph.pickled_data,
        )
        .join(graph_database.Graph)
        .filter(graph_database.GraphMeta.bytecode_id.in_(graph_ids_to_fetch))
      )

      ids_to_graphs = {
        bytecode_id: pickle.loads(data) for bytecode_id, data in query
      }

    if len(graph_ids_to_fetch) != len(ids_to_graphs):
      raise EnvironmentError(
        f"Graph IDs not found in database {self.graph_db.url}: "
        f"{set(graph_ids_to_fetch) - set(ids_to_graphs.keys())}"
      )

    # Encode the graphs
    ids_to_encoded_sequences = {}
    ids_to_grouping_ids = {}
    ids_to_node_masks = {}
    for bytecode_id, graph in ids_to_graphs.items():
      seqs, ids, node_mask = self.encode_graph(graph)
      ids_to_encoded_sequences[bytecode_id] = seqs
      ids_to_grouping_ids[bytecode_id] = ids
      ids_to_node_masks[bytecode_id] = node_mask

    return ids_to_encoded_sequences, ids_to_grouping_ids, ids_to_node_masks

  def EncodeBytecodes(self, bytecode_ids: typing.List[int]):
    with self.unlabelled_graph_db.Session() as session:
      # TODO(github.com/ChrisCummins/ProGraML/issues/20): Rebuild networkx from
      # graph tuples.
      query = (
        session.query(
          graph_database.GraphMeta.bytecode_id,
          graph_database.Graph.pickled_data,
        )
        .join(graph_database.Graph)
        .filter(graph_database.GraphMeta.bytecode_id.in_(bytecode_ids))
      )

      ids_to_graphs = {
        bytecode_id: pickle.loads(data) for bytecode_id, data in query
      }

    if len(bytecode_ids) != len(ids_to_graphs):
      raise EnvironmentError(
        f"Graph IDs not found in database {self.graph_db.url}: "
        f"{set(graph_ids_to_fetch) - set(ids_to_graphs.keys())}"
      )

    encoded_sequences, segment_ids, node_masks = [], [], []

    for bytecode_id in bytecode_ids:
      graph = ids_to_graphs[bytecode_id]
      seqs, ids, node_mask = self.encode_graph(graph)
      encoded_sequences.append(seqs)
      segment_ids.append(ids)
      node_masks.append(node_mask)

    return encoded_sequences, segment_ids, node_masks

  @property
  def unlabelled_graph_db(self) -> graph_database.Database:
    """Get the database of unlabelled graphs."""
    if self._unlabelled_graph_db:
      return self._unlabelled_graph_db
    elif FLAGS.unlabelled_graph_db:
      self._unlabelled_graph_db = FLAGS.unlabelled_graph_db()
      return self._unlabelled_graph_db
    else:
      raise app.UsageError("--unlabelled_graph_db must be set")

  def _EncodeStringsWithGroupings(
    self, strings: typing.List[str]
  ) -> typing.Tuple[np.array, np.array]:
    """Encode the given strings and return a flattened list of the encoded
    values, along with grouping IDs.
    """
    encoded_sequences = self.bytecode_encoder.EncodeBytecodeStrings(
      strings, pad=False
    )
    statement_indices = []
    for i, enc in enumerate(encoded_sequences):
      statement_indices.append([i] * len(enc))

    if not encoded_sequences == []:
      encoded_sequences = np.concatenate(encoded_sequences)
    if not statement_indices == []:
      statement_indices = np.concatenate(statement_indices)

    return encoded_sequences, np.array(statement_indices, dtype=np.int32)

  def _GraphToEncodedStatementGroups(
    self, graph: nx.MultiDiGraph
  ) -> typing.Tuple[np.array, np.array, np.array]:
    """Serialize the graph to an encoded sequence and set of statement indices.
    """
    serialized_node_list = list(cdfg.SerializeToStatementList(graph))
    statement_nodes = set(serialized_node_list)
    node_mask = np.array(
      [1 if node in statement_nodes else 0 for node in graph.nodes()],
      dtype=np.int32,
    )

    if not any(node_mask):
      return (
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
      )

    strings_to_encode = [
      graph.nodes[n].get("original_text", "") for n in serialized_node_list
    ]

    seqs, ids = self._EncodeStringsWithGroupings(strings_to_encode)

    if max(ids) > graph.number_of_nodes():
      app.Error(
        "Found max ID %s in graph of only %s nodes",
        max(ids),
        graph.number_of_nodes(),
      )

    return seqs, ids, node_mask

  def _GraphToEncodedIdentifierGroups(
    self, graph: nx.MultiDiGraph
  ) -> typing.Tuple[np.array, np.array, np.array]:
    """Serialize the graph to an encoded sequence and set of statement indices.
    """
    identifiers, node_mask = [], []
    for node, type_ in graph.nodes(data="type"):
      if type_ == "identifier":
        identifiers.append(node)
        node_mask.append(1)
      else:
        node_mask.append(0)
    node_mask = np.array(node_mask, dtype=np.int32)

    if not any(node_mask):
      return (
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
      )

    strings_to_encode = labtypes.flatten(
      [
        [
          graph.nodes[n].get("original_text", "")
          for n in graph_query.GetStatementsForNode(graph, identifier)
        ]
        for identifier in identifiers
      ]
    )

    seqs, ids = self._EncodeStringsWithGroupings(strings_to_encode)

    if max(ids) > graph.number_of_nodes():
      app.Error(
        "Found max ID %s in graph of only %s nodes",
        max(ids),
        graph.number_of_nodes(),
      )

    return seqs, ids, node_mask
