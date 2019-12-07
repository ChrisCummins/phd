"""This module defines a lexer interface as described in PACT'17 paper."""
import copy
import enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from deeplearning.clgen.proto import internal_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import progress

app.DEFINE_integer(
  "lexer_max_chunksize_mb",
  32,
  "The maximum number of megabytes of strings to feed into a single lexer "
  "invocation.",
)

FLAGS = app.FLAGS

# The native C++ lexer binary.
LEXER_BINARY = bazelutil.DataPath(
  "phd/deeplearning/clgen/corpuses/lexer/lexer_worker"
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
    max_chunksize: Optional[int] = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    self.candidate_tokens = {
      LexerType.LLVM: LLVM_TOKENS,
      LexerType.OPENCL: OPENCL_TOKENS,
    }[type]

    self.vocab = copy.deepcopy(initial_vocab)
    self.max_chunksize = max_chunksize or (
      FLAGS.lexer_max_chunksize_mb * 1024 * 1024
    )
    self.ctx = ctx

  def _Lex(self, texts: List[str]) -> Tuple[List[np.array], Dict[str, int]]:
    """Run lexer on a list of texts.

    Args:
      texts: The strings to lex.
      candidate_tokens: A list of candidate vocabulary words.
      vocab: A mapping from <word, encoded> value.

    Returns:
      A list of lists of shape (len(texts), encoded_length), where each element
      is an integer encoded token.
    """
    message = internal_pb2.LexerBatchJob(
      input=[internal_pb2.LexerJob(string=text) for text in texts],
      candidate_token=self.candidate_tokens,
      vocabulary=self.vocab,
    )
    pbutil.RunProcessMessageInPlace(
      [str(LEXER_BINARY)], message, timeout_seconds=3600
    )
    encoded = [np.array(j.token, dtype=np.int32) for j in message.input]
    vocabulary_out = dict(message.vocabulary)
    return encoded, vocabulary_out

  def LexAndUpdateVocab(self, texts: List[str],) -> List[np.array]:
    """Encode the given texts using the vocabulary.

    The vocabulary is lazily constructed. If a token is found that is not in the
    vocabulary, it is added.

    There is non-negligible overhead in calling this method. For the sake of
    efficiency try to minimize the number of calls to this method.

    Returns:
      A list of encoded texts.
    """
    token_count = 0
    with self.ctx.Profile(
      3,
      lambda t: f"Lexed {len(texts)} strings ({humanize.DecimalPrefix(token_count / t, ' tokens/sec')})",
    ):
      lexed = []
      strings_to_lex = []
      chunksize = 0

      for text in texts:
        if chunksize >= self.max_chunksize:
          chunk, self.vocab = self._Lex(strings_to_lex)
          lexed += chunk
          chunksize = 0
          strings_to_lex = []
        strings_to_lex.append(text)
        chunksize += len(text)

      if strings_to_lex:
        chunk, self.vocab = self._Lex(strings_to_lex)
        lexed += chunk

      # Used in profiling callback.
      token_count = sum([len(encoded) for encoded in lexed])

    return lexed

  @staticmethod
  def ClampVocab(encoded: np.array, max_vocab_element: int):
    """Clamp values to the range [0, max_vocab_element + 1].

    Use this method to set unknown elements in the vocab to a known value.
    """
    encoded[np.where(encoded > max_vocab_element)] = max_vocab_element + 1
    return encoded

  def Lex(self, texts: List[str],) -> List[np.array]:
    """Lex a list of strings.

    If any out-of-vocab elements appear, they are set with max(vocab) + 1
    values.

    There is non-negligible overhead in calling this method. For the sake of
    efficiency try to minimize the number of calls to this method.

    Args:
      texts: A list of strings to lex.

    Returns:
      A list of encoded sequences, where each element in an encoded sequence is
      in the range [0, max(vocab) + 1].
    """
    token_count = 0
    with self.ctx.Profile(
      3,
      lambda t: f"Lexed {len(texts)} strings ({humanize.DecimalPrefix(token_count / t, ' tokens/sec')})",
    ):
      max_vocab_element = len(self.vocab) - 1

      lexed = []
      strings_to_lex = []
      chunksize = 0

      for text in texts:
        if chunksize >= self.max_chunksize:
          chunk, vocab = self._Lex(strings_to_lex)
          if len(vocab) > len(self.vocab):
            chunk = [self.ClampVocab(x, max_vocab_element) for x in chunk]
          lexed += chunk
          chunksize = 0
          strings_to_lex = []
        strings_to_lex.append(text)
        chunksize += len(text)

      if strings_to_lex:
        chunk, vocab = self._Lex(strings_to_lex)
        if len(vocab) > len(self.vocab):
          chunk = [self.ClampVocab(x, max_vocab_element) for x in chunk]
        lexed += chunk

      # Used in profiling callback.
      token_count = sum([len(encoded) for encoded in lexed])

    return lexed
