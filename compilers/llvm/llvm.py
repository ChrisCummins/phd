"""Shared code for LLVM modules."""
from absl import flags


FLAGS = flags.FLAGS


class LlvmError(EnvironmentError):
  pass


class LlvmTimeout(LlvmError):
  pass
