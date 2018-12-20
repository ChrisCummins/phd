"""Shared code for LLVM modules."""
import typing

from absl import flags


FLAGS = flags.FLAGS


class LlvmError(EnvironmentError):

  def __init__(self, msg: typing.Optional[str] = None,
               returncode: typing.Optional[int] = None,
               stderr: typing.Optional[str] = None,
               command: typing.Optional[typing.List[str]] = None):
    self.msg = msg
    self.returncode = returncode
    self.stderr = stderr
    self.command = command

  def __repr__(self):
    return str(self.msg or type(self).__name__)


class LlvmTimeout(LlvmError):
  pass
