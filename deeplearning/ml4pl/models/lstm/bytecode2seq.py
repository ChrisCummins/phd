"""Module to convert bytecode IDs into vocabulary sequences."""
import json
import typing

import keras
import numpy as np
import sqlalchemy as sql

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.clgen.proto import internal_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ncc import vocabulary as inst2vec_vocab
from deeplearning.ncc.inst2vec import api as inst2vec
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import pbutil

app.DEFINE_database(
  "bytecode_db",
  bytecode_database.Database,
  None,
  "URL of database to read bytecodes from.",
  must_exist=True,
)

app.DEFINE_input_path(
  "bytecode_vocabulary",
  bazelutil.DataPath("phd/deeplearning/ml4pl/models/lstm/llvm_vocab.json"),
  "Override the default LLVM vocabulary file. Use "
  "//deeplearning/ml4pl/models/lstm:derive_vocabulary to generate a "
  "vocabulary.",
)

app.DEFINE_integer(
  "max_encoded_length",
  None,
  "Override the max_encoded_length value loaded from the vocabulary.",
)

FLAGS = app.FLAGS

LEXER_WORKER = bazelutil.DataPath(
  "phd/deeplearning/clgen/corpuses/lexer/lexer_worker"
)

LLVM_IR_TOKENS = [
  # Primitive types
  "void",
  "half",
  "float",
  "double",
  "fp128",
  "x86_fp80",
  "ppc_fp128",
  # Note that ints can have any bit width up to 2^23 - 1, so this is
  # non-exhaustive.
  "i32",
  "i64",
  "i128",
  # Terminator ops
  "ret",
  "br",
  "switch",
  "indirectbr",
  "invoke",
  "callbr",
  "resume",
  "catchswitch",
  "catchret",
  "cleanupret",
  "unreachable",
  "unreachable",
  # Unary ops
  "fneg",
  # Binary ops
  "add",
  "fadd",
  "sub",
  "fsub",
  "mul",
  "fmul",
  "udiv",
  "sdiv",
  "fdiv",
  "urem",
  "srem",
  "frem",
  # Bitwise binary ops
  "shl",
  "lshr",
  "ashr",
  "and",
  "or",
  "xor",
  # Vector ops
  "extractelement",
  "insertelement",
  "shufflevector",
  # Aggregate ops
  "extractvalue",
  "insertvalue",
  # Memory Access
  "alloca",
  "load",
  "store",
  "fence",
  "cmpxchg",
  "atomicrmw",
  "getelementptr",
  # Conversion ops
  "trunc",
  "to",
  "zext",
  "sext",
  "fptrunc",
  "fpext",
  "fptoui",
  "fptose",
  "uitofp",
  "sitofp",
  "ptrtoint",
  "inttoptr",
  "bitcast",
  "addrspacecast",
  # Other ops
  "icmp",
  "fcmp",
  "phi",
  "select",
  "call",
  "va_arg",
  "landingpad",
  "catchpad",
  "cleanuppad",
  # Misc keywords
  "define",
  "declare",
  "private",
  "unnamed_addr",
  "constant",
  "nounwind",
  "nocapture",
]

# The vocabulary used in PACT'17 work.
OPENCL_TOKENS = [
  "__assert",
  "__attribute",
  "__builtin_astype",
  "__clc_fabs",
  "__clc_fma",
  "__constant",
  "__global",
  "__inline",
  "__kernel",
  "__local",
  "__private",
  "__read_only",
  "__read_write",
  "__write_only",
  "abs",
  "alignas",
  "alignof",
  "atomic_add",
  "auto",
  "barrier",
  "bool",
  "break",
  "case",
  "char",
  "clamp",
  "complex",
  "const",
  "constant",
  "continue",
  "default",
  "define",
  "defined",
  "do",
  "double",
  "elif",
  "else",
  "endif",
  "enum",
  "error",
  "event_t",
  "extern",
  "fabs",
  "false",
  "float",
  "for",
  "get_global_id",
  "get_global_size",
  "get_local_id",
  "get_local_size",
  "get_num_groups",
  "global",
  "goto",
  "half",
  "if",
  "ifdef",
  "ifndef",
  "image1d_array_t",
  "image1d_buffer_t",
  "image1d_t",
  "image2d_array_t",
  "image2d_t",
  "image3d_t",
  "imaginary",
  "include",
  "inline",
  "int",
  "into",
  "kernel",
  "line",
  "local",
  "long",
  "noreturn",
  "pragma",
  "private",
  "quad",
  "read_only",
  "read_write",
  "register",
  "restrict",
  "return",
  "sampler_t",
  "short",
  "shuffle",
  "signed",
  "size_t",
  "sizeof",
  "sqrt",
  "static",
  "struct",
  "switch",
  "true",
  "typedef",
  "u32",
  "uchar",
  "uint",
  "ulong",
  "undef",
  "union",
  "unsigned",
  "void",
  "volatile",
  "while",
  "wide",
  "write_only",
]


def Encode(
  bytecodes: typing.List[str],
  vocab: typing.Dict[str, int],
  language: str = "llvm",
) -> typing.Tuple[typing.List[typing.List[int]], typing.Dict[str, int]]:
  """Encode the given bytecodes using the vocabulary.

  The vocabulary is lazily constructed. If a token is found that is not in the
  vocabulary, it is added.

  There is non-negligible overhead in calling this method. For the sake of
  efficiency try to minimize the number of calls to this method.

  Returns:
    A list of encoded bytecodes, and the output vocabulary.
  """
  tokens = {"llvm": LLVM_IR_TOKENS, "opencl": OPENCL_TOKENS,}[language]

  message = internal_pb2.LexerBatchJob(
    input=[internal_pb2.LexerJob(string=s) for s in bytecodes],
    candidate_token=tokens,
    vocabulary=vocab,
  )
  pbutil.RunProcessMessageInPlace([LEXER_WORKER], message, timeout_seconds=3600)
  return [list(j.token) for j in message.input], dict(message.vocabulary)


def EncodeWithFixedVocab(
  bytecodes: typing.List[str],
  vocab: typing.Dict[str, int],
  language: str = "llvm",
) -> typing.List[typing.List[int]]:
  encoded_sequences, vocab_out = Encode(bytecodes, vocab, language)
  if len(vocab_out) != len(vocab):
    app.Error(
      "Encoded vocabulary has different size "
      f"({len(vocab_out)}) than the input "
      f"({len(vocab)})"
    )
  return encoded_sequences


class EncoderBase(object):
  """Base class for implementing bytecode encoders."""

  def __init__(self):
    self._bytecode_db = None

  def Encode(
    self, bytecode_ids: typing.List[int]
  ) -> typing.List[typing.List[int]]:
    """Convert a list of bytecode IDs to a list of encoded sequences."""
    raise NotImplementedError("abstract class")

  @property
  def bytecode_db(self) -> bytecode_database.Database:
    """Get the bytecode database."""
    if self._bytecode_db:
      return self._bytecode_db
    elif FLAGS.bytecode_db:
      self._bytecode_db = FLAGS.bytecode_db()
      return self._bytecode_db
    else:
      raise app.UsageError("--bytecode_db must be set")


class BytecodeEncoder(EncoderBase):
  def __init__(self):
    super(BytecodeEncoder, self).__init__()

    # Load the vocabulary used for encoding LLVM bytecode.
    with open(FLAGS.bytecode_vocabulary) as f:
      data_to_load = json.load(f)
    self.vocabulary = data_to_load["vocab"]
    self.max_sequence_length = data_to_load["max_encoded_length"]
    self.language = "llvm"

    # Allow the --max_encoded_length to override the value stored in the
    # vocabulary file.
    if FLAGS.max_encoded_length:
      app.Log(
        1,
        "Changing max sequence length from %s to %s",
        self.max_sequence_length,
        FLAGS.max_encoded_length,
      )
      self.max_sequence_length = FLAGS.max_encoded_length

    # The out-of-vocabulary padding value, used to pad sequences to the same
    # length.
    self.pad_val = len(self.vocabulary)
    assert self.pad_val not in self.vocabulary

    app.Log(
      1,
      "Bytecode encoder using %s-element vocabulary with maximum "
      "sequence length %s",
      self.vocabulary_size_with_padding_token,
      self.max_sequence_length,
    )

  @property
  def vocabulary_size_with_padding_token(self) -> int:
    return len(self.vocabulary) + 1

  def Encode(self, bytecode_ids: typing.List[int]) -> np.array:
    with self.bytecode_db.Session() as session:
      query = session.query(
        bytecode_database.LlvmBytecode.id,
        bytecode_database.LlvmBytecode.bytecode,
      )
      query = query.filter(bytecode_database.LlvmBytecode.id.in_(bytecode_ids))

      bytecode_id_to_string = {
        bytecode_id: bytecode for bytecode_id, bytecode in query
      }
      if len(set(bytecode_ids)) != len(bytecode_id_to_string):
        raise EnvironmentError(
          f"len(bytecode_ids)={len(bytecode_ids)} != "
          f"len(bytecode_id_to_string)={len(bytecode_id_to_string)}"
        )

    return self.EncodeBytecodeStrings(
      [bytecode_id_to_string[bytecode_id] for bytecode_id in bytecode_ids]
    )

  def EncodeBytecodeStrings(self, strings: typing.List[str], pad: bool = True):
    # Encode the requested bytecodes.
    encoded_sequences = EncodeWithFixedVocab(
      strings, self.vocabulary, self.language
    )
    if len(strings) != len(encoded_sequences):
      raise EnvironmentError(
        f"len(strings)={len(strings)} != "
        f"len(encoded_sequences)={len(encoded_sequences)}"
      )
    if pad:
      return np.array(
        keras.preprocessing.sequence.pad_sequences(
          encoded_sequences, maxlen=self.max_sequence_length, value=self.pad_val
        )
      )
    else:
      return np.array(encoded_sequences)


class Inst2VecEncoder(BytecodeEncoder):
  """Translate bytecode IDs to inst2vec encoded sequences."""

  # TODO(github.com/ChrisCummins/ProGraML/issues/20): There is no need to
  # inherit from BytecodeEncoder, and this causes confusion with having to set
  # the max_sequence_length twice. Refactor.

  def __init__(self):
    self.vocab = inst2vec_vocab.VocabularyZipFile.CreateFromPublishedResults()

    # Unpack the vocabulary zipfile.
    self.vocab.__enter__()

    self.pad_val = len(self.vocab.dictionary)
    assert self.pad_val not in self.vocab.dictionary.values()

    # We must call the superclass constructor *after* unpacking the vocabulary
    # zipfile.
    super(Inst2VecEncoder, self).__init__()

    with self.bytecode_db.Session() as session:
      max_linecount = session.query(
        sql.func.max(bytecode_database.LlvmBytecode.linecount)
      ).one()[0]
      self.max_sequence_length = max_linecount

    # Allow the --max_encoded_length to override the value stored in the
    # vocabulary file.
    if FLAGS.max_encoded_length:
      app.Log(
        1,
        "Changing inst2vec max sequence length from %s to %s",
        self.max_sequence_length,
        FLAGS.max_encoded_length,
      )
      self.max_sequence_length = FLAGS.max_encoded_length

  def __del__(self):
    # Tidy up the unpacked vocabulary zipfile.
    self.vocab.__exit__()

  def EncodeBytecodeStrings(self, strings: typing.List[str]):
    with self.vocab as vocab:
      encoded_sequences = [
        inst2vec.EncodeLlvmBytecode(bytecode, vocab) for bytecode in strings
      ]

    return np.array(
      keras.preprocessing.sequence.pad_sequences(
        encoded_sequences, maxlen=self.max_sequence_length, value=self.pad_val,
      )
    )

  @property
  def vocabulary_size_with_padding_token(self) -> int:
    return len(self.vocab.dictionary) + 1


class OpenClEncoder(EncoderBase):
  """Translate bytecode IDs to encoded OpenCL sources.

  This pre-computes the encoded sequences for all values during construction
  time.
  """

  def __init__(self):
    super(OpenClEncoder, self).__init__()

    # Map relpath -> src.
    df = make_devmap_dataset.MakeGpuDataFrame(
      opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df,
      "amd_tahiti_7970",
    )
    relpath_to_src = {
      row["relpath"]: row["program:opencl_src"] for _, row in df.iterrows()
    }

    # Map relpath -> bytecode ID.
    with self.bytecode_db.Session() as session:
      query = session.query(
        bytecode_database.LlvmBytecode.id,
        bytecode_database.LlvmBytecode.relpath,
      )
      query = query.filter(
        bytecode_database.LlvmBytecode.source_name == "pact17_opencl_devmap"
      )
      relpath_to_bytecode_id = {
        relpath: bytecode_id for bytecode_id, relpath in query
      }

    not_found = set(relpath_to_src.keys()) - set(relpath_to_bytecode_id.keys())
    if not_found:
      raise OSError(f"Relpaths not bound in bytecode database: {not_found}")

    # Map bytecode ID -> OpenCL.
    bytecode_id_src_pairs = {
      (relpath_to_bytecode_id[relpath], src)
      for relpath, src in relpath_to_src.items()
    }

    encoded, self.vocabulary = Encode(
      [x[1] for x in bytecode_id_src_pairs], {}, "opencl"
    )

    self.max_sequence_length = max(len(m) for m in encoded)
    # Allow the --max_encoded_length to override the value stored in the
    # vocabulary file.
    if FLAGS.max_encoded_length:
      app.Log(
        1,
        "Changing max sequence length from %s to %s",
        self.max_sequence_length,
        FLAGS.max_encoded_length,
      )
      self.max_sequence_length = FLAGS.max_encoded_length

    self.bytecode_to_encoded = {
      bytecode_id: encoded
      for (bytecode_id, src), encoded in zip(bytecode_id_src_pairs, encoded)
    }

    self.pad_val = len(self.vocabulary)
    assert self.pad_val not in self.vocabulary.values()

  @property
  def vocabulary_size_with_padding_token(self) -> int:
    return len(self.vocabulary) + 1

  def Encode(self, bytecode_ids: typing.List[int]):
    encoded_sequences = [
      self.bytecode_to_encoded[bytecode_id] for bytecode_id in bytecode_ids
    ]
    return np.array(
      keras.preprocessing.sequence.pad_sequences(
        encoded_sequences, maxlen=self.max_sequence_length, value=self.pad_val,
      )
    )
