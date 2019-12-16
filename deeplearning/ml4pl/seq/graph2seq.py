"""Module for conversion from unlabelled graphs to encoded sequences."""
import json
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import lru
import numpy as np
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.seq import graph2seq_pb2
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import progress

FLAGS = app.FLAGS

app.DEFINE_integer(
  "graph2seq_cache_entries",
  512000,
  "The number of ID -> encoded sequence entries to cache.",
)

GRAPH_ENCODER_WORKER = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/seq/graph_encoder_worker"
)

# The vocabulary to use for LLVM encoders. Use
# //deeplearning/ml4pl/seq:derive_vocab to generate a vocabulary.
LLVM_VOCAB = bazelutil.DataPath("phd/deeplearning/ml4pl/seq/llvm_vocab.json")


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
  ) -> List[Union[np.array, graph2seq_pb2.ProgramGraphSeq]]:
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
      graphs: A list of unlabelled graphs to encode.
      ctx: A logging context.

    Returns:
      A list of encoded sequences.
    """
    unknown_ir_ids = set(
      graph.ir_id
      for graph in graphs
      if graph.ir_id not in self.ir_id_to_encoded
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


class StatementEncoder(EncoderBase):
  """Encode graphs to per-node sub-sequences.

  This uses the graph structure to produce a tokenized sequence ordered by
  depth first traversal, and allows mapping from graph nodes to sub-seqeuences
  within the encoded output.
  """

  def __init__(
    self,
    graph_db: graph_tuple_database.Database,
    proto_db: unlabelled_graph_database.Database,
    ir2seq_encoder: ir2seq.EncoderBase,
    cache_size: Optional[int] = None,
  ):
    super(StatementEncoder, self).__init__(graph_db, ir2seq_encoder, cache_size)
    self.proto_db = proto_db

    with open(LLVM_VOCAB) as f:
      data_to_load = json.load(f)
    self.vocabulary = data_to_load["vocab"]

  def Encode(
    self,
    graphs: List[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[graph2seq_pb2.ProgramGraphSeq]:
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
      ctx: A logging context.

    Returns:
      A list of EncodedSubsequence tuples, where each tuple maps a graph to
      encoded sequences, subsequence groupings, and node_mask arrays which list
      the nodes which are selected from each graph.
    """
    # Look for any graphs that are unknown.
    graph_ids_to_encode: Set[int] = {
      graph.ir_id
      for graph in graphs
      if graph.ir_id not in self.ir_id_to_encoded
    }

    if graph_ids_to_encode:
      sorted_graph_ids_to_encode = sorted(graph_ids_to_encode)

      # Fetch the protos for the graphs that we need to encode.
      with self.proto_db.Session() as session:
        sorted_protos_to_encode = [
          row.proto
          for row in session.query(unlabelled_graph_database.ProgramGraph)
          .options(
            sql.orm.joinedload(unlabelled_graph_database.ProgramGraph.data)
          )
          .filter(
            unlabelled_graph_database.ProgramGraph.ir_id.in_(
              graph_ids_to_encode
            )
          )
          .order_by(unlabelled_graph_database.ProgramGraph.ir_id)
        ]
        if len(sorted_protos_to_encode) != len(sorted_graph_ids_to_encode):
          raise OSError(
            f"Requested {len(sorted_graph_ids_to_encode)} protos "
            "from database but received "
            f"{len(sorted_protos_to_encode)}"
          )

      # Encode the unknown graphs.
      sorted_encoded = self.EncodeGraphs(sorted_protos_to_encode, ctx=ctx)
      # Encode and cache the unknown graphs.
      for ir_id, encoded in zip(sorted_graph_ids_to_encode, sorted_encoded):
        self.ir_id_to_encoded[ir_id] = encoded

    return [self.ir_id_to_encoded[graph.ir_id] for graph in graphs]

  def EncodeGraphs(
    self,
    graphs: List[programl_pb2.ProgramGraph],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[graph2seq_pb2.ProgramGraphSeq]:
    """Encode a list of graphs and return them in order.

    Args:
      A list of zero or more strings.

    Returns:
      A pair of <encoded_sequences, statement_indices> arrays.
    """
    with ctx.Profile(
      3,
      lambda t: (
        f"Encoded {len(graphs)} graphs "
        f"({humanize.DecimalPrefix(token_count / t, ' tokens/sec')}"
      ),
    ):
      message = graph2seq_pb2.GraphEncoderJob(
        vocabulary=self.vocabulary, graph=graphs,
      )
      pbutil.RunProcessMessageInPlace(
        [str(GRAPH_ENCODER_WORKER)], message, timeout_seconds=3600
      )
      encoded_graphs = [encoded for encoded in message.seq]
      token_count = sum(len(encoded.encoded) for encoded in encoded_graphs)
      if len(encoded_graphs) != len(graphs):
        raise ValueError(
          f"Requested {len(graphs)} graphs to be encoded but "
          f"received {len(encoded_graphs)}"
        )

    return encoded_graphs
