"""Module to convert intermediate representations into vocabulary sequences."""
import json
from typing import List
from typing import Tuple

import numpy as np
import sqlalchemy as sql

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.seq import lexers
from deeplearning.ncc import vocabulary as inst2vec_vocab
from deeplearning.ncc.inst2vec import api as inst2vec
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize
from labm8.py import progress


FLAGS = app.FLAGS

# The vocabulary to use for LLVM encoders. Use
# //deeplearning/ml4pl/seq:derive_vocab to generate a vocabulary.
LLVM_VOCAB = bazelutil.DataPath("phd/deeplearning/ml4pl/seq/llvm_vocab.json")


class EncoderBase(object):
  """Base class for implementing bytecode encoders."""

  def __init__(
    self, ir_db: ir_database.Database,
  ):
    self.ir_db = ir_db

  def Encode(
    self, ids: List[int], ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[np.array]:
    """Convert a list of IR IDs to a list of encoded sequences."""
    raise NotImplementedError("abstract class")

  @property
  def vocabulary_size(self) -> int:
    """Get the size of the vocabulary, including the unknown-vocab element."""
    raise NotImplementedError("abstract class")

  @property
  def max_encoded_length(self) -> int:
    """Return an upper bound on the length of the encoded sequences."""
    raise NotImplementedError("abstract class")


class LlvmEncoder(EncoderBase):
  """An encoder for LLVM intermediate representations."""

  def __init__(self, *args, **kwargs):
    super(LlvmEncoder, self).__init__(*args, **kwargs)

    # Load the vocabulary used for encoding LLVM bytecode.
    with open(LLVM_VOCAB) as f:
      data_to_load = json.load(f)
    vocab = data_to_load["vocab"]
    self._max_encoded_length = data_to_load["max_encoded_length"]

    self.lexer = lexers.Lexer(type=lexers.LexerType.LLVM, initial_vocab=vocab)

  def Encode(
    self, ids: List[int], ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[np.array]:
    """Encode a list of IR IDs.

    Args:
      ids: Intermediate representation IDs.

    Returns:
      A list of encoded sequences.
    """
    sorted_unique_ids: List[int] = list(sorted(set(ids)))
    with self.ir_db.Session() as session:
      sorted_unique_ir: List[str] = [
        ir_database.IntermediateRepresentation.DecodeBinaryIr(row.binary_ir)
        for row in session.query(
          ir_database.IntermediateRepresentation.binary_ir,
        )
        .filter(
          ir_database.IntermediateRepresentation.id.in_(ids),
          ir_database.IntermediateRepresentation.compilation_succeeded == True,
        )
        .order_by(ir_database.IntermediateRepresentation.id)
      ]

      if len(sorted_unique_ir) != len(sorted_unique_ids):
        raise KeyError(
          f"Requested {len(sorted_unique_ids)} IRs from database but received "
          f"{len(sorted_unique_ir)}"
        )

      sorted_unique_encodeds: List[np.array] = self.lexer.Lex(
        sorted_unique_ir, ctx=ctx
      )

      id_to_encoded = {
        id: encoded
        for id, encoded in zip(sorted_unique_ids, sorted_unique_encodeds)
      }

    return [id_to_encoded[id] for id in ids]

  @property
  def vocabulary_size(self) -> int:
    """Return the size of the encoder vocabulary."""
    return self.lexer.vocabulary_size

  @property
  def max_encoded_length(self) -> int:
    """Return an upper bound on the length of the encoded sequences."""
    return self._max_encoded_length


class OpenClEncoder(EncoderBase):
  """An OpenCL source-level encoder.

  This pre-computes the encoded sequences for all values during construction
  time.
  """

  def __init__(self, *args, **kwargs):
    super(OpenClEncoder, self).__init__(*args, **kwargs)

    # We start with an empty vocabulary and build it from inputs.
    self.lexer = lexers.Lexer(type=lexers.LexerType.OPENCL, initial_vocab={})

    # Map relpath -> src.
    df = make_devmap_dataset.MakeGpuDataFrame(
      opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df,
      "amd_tahiti_7970",
    )
    relpath_to_src = {
      row["relpath"]: row["program:opencl_src"] for _, row in df.iterrows()
    }

    # Map relpath -> bytecode ID.
    with self.ir_db.Session() as session:
      relpath_to_id = {
        row.relpath: row.id
        for row in session.query(
          ir_database.IntermediateRepresentation.id,
          ir_database.IntermediateRepresentation.relpath,
        ).filter(
          ir_database.IntermediateRepresentation.source_language
          == ir_database.SourceLanguage.OPENCL,
          ir_database.IntermediateRepresentation.compilation_succeeded == True,
          ir_database.IntermediateRepresentation.source
          == "pact17_opencl_devmap",
          ir_database.IntermediateRepresentation.relpath.in_(
            relpath_to_src.keys()
          ),
        )
      }

    not_found = set(relpath_to_src.keys()) - set(relpath_to_id.keys())
    if not_found:
      raise OSError(
        f"{humanize.Plural(len(not_found), 'OpenCL relpath')} not"
        " found in IR database"
      )

    # Encode the OpenCL sources.
    sorted_id_src_pairs: List[Tuple[int, str]] = {
      (relpath_to_id[relpath], relpath_to_src[relpath])
      for relpath in sorted(relpath_to_src.keys())
    }
    sorted_encodeds: List[np.array] = self.lexer.Lex(
      [src for id, src in sorted_id_src_pairs]
    )

    self._max_encoded_length = max(len(encoded) for encoded in sorted_encodeds)

    # Map id -> encoded.
    self.id_to_encoded = {
      id: encoded
      for (id, _), encoded in zip(sorted_id_src_pairs, sorted_encodeds)
    }

  def Encode(
    self, ids: List[int], ctx: progress.ProgressContext = progress.NullContext
  ) -> List[np.array]:
    """Encode a list of IR IDs.

    Args:
      ids: Intermediate representation IDs.

    Returns:
      A list of encoded OpenCL sequences.
    """
    return [self.id_to_encoded[id] for id in ids]

  @property
  def vocabulary_size(self) -> int:
    """Return the size of the encoder vocabulary."""
    return self.lexer.vocabulary_size

  @property
  def max_encoded_length(self) -> int:
    """Return an upper bound on the length of the encoded sequences."""
    return self._max_encoded_length


class Inst2VecEncoder(EncoderBase):
  """Translate intermediate representations to inst2vec encoded sequences."""

  def __init__(self, *args, **kwargs):
    super(Inst2VecEncoder, self).__init__(*args, **kwargs)

    self.vocab = inst2vec_vocab.VocabularyZipFile.CreateFromPublishedResults()

    # Unpack the vocabulary zipfile.
    self.vocab.__enter__()

    # inst2vec encodes one element per line of IR.
    with self.ir_db.Session() as session:
      max_line_count = session.query(
        sql.func.max(ir_database.IntermediateRepresentation.line_count)
      ).scalar()
      self._max_encoded_length = max_line_count

  def __del__(self):
    # Tidy up the unpacked vocabulary zipfile.
    self.vocab.__exit__()

  def Encode(
    self, ids: List[int], ctx: progress.ProgressContext = progress.NullContext,
  ):
    """Encode a list of IR IDs.

    Args:
      ids: Intermediate representation IDs.

    Returns:
      A list of encoded OpenCL sequences.
    """
    sorted_unique_ids: List[int] = list(sorted(set(ids)))
    with self.ir_db.Session() as session:
      sorted_unique_ir: List[str] = [
        ir_database.IntermediateRepresentation.DecodeBinaryIr(row.binary_ir)
        for row in session.query(
          ir_database.IntermediateRepresentation.binary_ir,
        )
        .filter(
          ir_database.IntermediateRepresentation.id.in_(ids),
          ir_database.IntermediateRepresentation.compilation_succeeded == True,
        )
        .order_by(ir_database.IntermediateRepresentation.id)
      ]

      if len(sorted_unique_ir) != len(sorted_unique_ids):
        raise KeyError(
          f"Requested {len(sorted_unique_ids)} IRs from database but received "
          f"{len(sorted_unique_ir)}"
        )

      token_count = 0
      with ctx.Profile(
        3,
        lambda t: (
          f"Encoded {len(sorted_unique_ir)} strings "
          f"({humanize.DecimalPrefix(token_count / t, ' tokens/sec')})"
        ),
      ):
        sorted_unique_encodeds: List[np.array] = [
          np.array(inst2vec.EncodeLlvmBytecode(ir, self.vocab), dtype=np.int32)
          for ir in sorted_unique_ir
        ]
        token_count = sum(len(encoded) for encoded in sorted_unique_encodeds)

      id_to_encoded = {
        id: encoded
        for id, encoded in zip(sorted_unique_ids, sorted_unique_encodeds)
      }

    return [id_to_encoded[id] for id in ids]

  @property
  def vocabulary_size(self) -> int:
    """Get the size of the vocabulary."""
    return len(self.vocab.dictionary)

  @property
  def max_encoded_length(self) -> int:
    """Return an upper bound on the length of the encoded sequences."""
    return self._max_encoded_length
