"""Module to convert LLVM IR into vocabulary sequences."""
import typing

import numpy as np
from labm8 import app
from labm8 import bazelutil
from labm8 import pbutil

from deeplearning.clgen.proto import internal_pb2

FLAGS = app.FLAGS

LEXER_WORKER = bazelutil.DataPath(
    'phd/deeplearning/clgen/corpuses/lexer/lexer_worker')

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


def Encode(bytecodes: typing.List[str], vocab: typing.Dict[str, int]
          ) -> typing.Tuple[typing.List[np.array], typing.Dict[str, int]]:
  """Encode the given bytecodes using the vocabulary.

  The vocabulary is lazily constructed. If a token is found that is not in the
  vocabulary, it is added.

  Returns:
    A list of encoded bytecodes, and the output vocabulary.
  """
  message = internal_pb2.LexerBatchJob(
      input=[internal_pb2.LexerJob(string=s) for s in bytecodes],
      candidate_token=LLVM_IR_TOKENS,
      vocabulary=vocab,
  )
  pbutil.RunProcessMessageInPlace([LEXER_WORKER], message, timeout_seconds=3600)
  return ([list(j.token) for j in message.input], dict(message.vocabulary))
