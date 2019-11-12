"""Module to convert LLVM IR into vocabulary sequences."""
import typing

import numpy as np

from deeplearning.clgen.proto import internal_pb2
from labm8 import app
from labm8 import bazelutil
from labm8 import pbutil

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

# The vocabulary used in PACT'17 work.
OPENCL_TOKENS = [
    '__assert',
    '__attribute',
    '__builtin_astype',
    '__clc_fabs',
    '__clc_fma',
    '__constant',
    '__global',
    '__inline',
    '__kernel',
    '__local',
    '__private',
    '__read_only',
    '__read_write',
    '__write_only',
    'abs',
    'alignas',
    'alignof',
    'atomic_add',
    'auto',
    'barrier',
    'bool',
    'break',
    'case',
    'char',
    'clamp',
    'complex',
    'const',
    'constant',
    'continue',
    'default',
    'define',
    'defined',
    'do',
    'double',
    'elif',
    'else',
    'endif',
    'enum',
    'error',
    'event_t',
    'extern',
    'fabs',
    'false',
    'float',
    'for',
    'get_global_id',
    'get_global_size',
    'get_local_id',
    'get_local_size',
    'get_num_groups',
    'global',
    'goto',
    'half',
    'if',
    'ifdef',
    'ifndef',
    'image1d_array_t',
    'image1d_buffer_t',
    'image1d_t',
    'image2d_array_t',
    'image2d_t',
    'image3d_t',
    'imaginary',
    'include',
    'inline',
    'int',
    'into',
    'kernel',
    'line',
    'local',
    'long',
    'noreturn',
    'pragma',
    'private',
    'quad',
    'read_only',
    'read_write',
    'register',
    'restrict',
    'return',
    'sampler_t',
    'short',
    'shuffle',
    'signed',
    'size_t',
    'sizeof',
    'sqrt',
    'static',
    'struct',
    'switch',
    'true',
    'typedef',
    'u32',
    'uchar',
    'uint',
    'ulong',
    'undef',
    'union',
    'unsigned',
    'void',
    'volatile',
    'while',
    'wide',
    'write_only',
]


def Encode(bytecodes: typing.List[str],
           vocab: typing.Dict[str, int],
           language: str = 'llvm'
          ) -> typing.Tuple[typing.List[np.array], typing.Dict[str, int]]:
  """Encode the given bytecodes using the vocabulary.

  The vocabulary is lazily constructed. If a token is found that is not in the
  vocabulary, it is added.

  There is non-negligible overhead in calling this method. For the sake of
  efficiency try to minimize the number of calls to this method.

  Returns:
    A list of encoded bytecodes, and the output vocabulary.
  """
  tokens = {
      'llvm': LLVM_IR_TOKENS,
      'opencl': OPENCL_TOKENS,
  }[language]

  message = internal_pb2.LexerBatchJob(
      input=[internal_pb2.LexerJob(string=s) for s in bytecodes],
      candidate_token=tokens,
      vocabulary=vocab,
  )
  pbutil.RunProcessMessageInPlace([LEXER_WORKER], message, timeout_seconds=3600)
  return ([list(j.token) for j in message.input], dict(message.vocabulary))
