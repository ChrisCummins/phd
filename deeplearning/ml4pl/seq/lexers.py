"""This module defines a lexer interface as described in PACT'17 paper."""
import copy
import enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from deeplearning.ml4pl.seq import ir2seq_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import progress

app.DEFINE_integer(
  "lexer_chunk_size_mb",
  32,
  "The maximum number of megabytes of strings to feed into a single lexer "
  "invocation.",
)

FLAGS = app.FLAGS

# The native C++ string encoder binary.
STRING_ENCODER_WORKER = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/seq/string_encoder_worker"
)


class LexerType(enum.Enum):
  """The type of lexer."""

  OPENCL = 1
  LLVM = 2


# The OpenCL vocabulary used in PACT'17 work.
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

# Tokens of LLVM intermediate representation.
LLVM_TOKENS = [
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


class Lexer(object):
  """A lexer."""

  def __init__(
    self,
    type: LexerType,
    initial_vocab: Dict[str, int],
    max_chunk_size: Optional[int] = None,
  ):
    self.candidate_tokens = {
      LexerType.LLVM: LLVM_TOKENS,
      LexerType.OPENCL: OPENCL_TOKENS,
    }[type]

    self.vocab = copy.deepcopy(initial_vocab)
    self.max_chunk_size = max_chunk_size or (
      FLAGS.lexer_chunk_size_mb * 1024 * 1024
    )

  @property
  def vocabulary_size(self) -> int:
    """Get the size of the vocabulary."""
    return len(self.vocab)

  @staticmethod
  def ClampVocab(encoded: np.array, max_vocab_element: int):
    """Clamp values to the range [0, max_vocab_element + 1].

    Use this method to set unknown elements in the vocab to a known value.
    """
    encoded[np.where(encoded > max_vocab_element)] = max_vocab_element + 1
    return encoded

  def Lex(
    self,
    texts: List[str],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[np.array]:
    """Lex a list of strings.

    If any out-of-vocab elements appear, they are set with max(vocab) + 1
    values.

    There is non-negligible overhead in calling this method. For the sake of
    efficiency try to minimize the number of calls.

    Args:
      texts: A list of strings to lex.
      ctx: A logging context.

    Returns:
      A list of lists of shape (len(texts), encoded_length), where each element
      is an integer encoded token in the range [0, self.vocabulary_size].
    """
    token_count = 0
    with ctx.Profile(
      3,
      lambda t: (
        f"Lexed {len(texts)} strings "
        f"({humanize.DecimalPrefix(token_count / t, ' tokens/sec')})"
      ),
    ):
      message = ir2seq_pb2.StringEncoderJob(
        string=texts, vocabulary=self.vocab,
      )
      pbutil.RunProcessMessageInPlace(
        [str(STRING_ENCODER_WORKER)], message, timeout_seconds=60
      )

      # Used in profiling callback.
      token_count = sum([len(seq.encoded) for seq in message.seq])

    encoded = [np.array(j.encoded, dtype=np.int32) for j in message.seq]
    if len(encoded) != len(texts):
      raise OSError(
        f"Lexer returned {len(texts)} sequences for {len(encoded)} inputs"
      )

    return encoded
