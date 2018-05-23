"""Custom exception hierachy used by CLgen."""


class CLgenError(Exception):
  """Top level error. Never directly thrown."""
  pass


class InternalError(CLgenError):
  """An internal module error.

  This class of errors should not leak outside of the module into user code.
  """
  pass


class UserError(CLgenError):
  """Raised in case of bad user interaction, e.g. an invalid argument."""
  pass


class File404(InternalError):
  """Data not found."""
  pass


class InvalidFile(UserError):
  """Raised in case a file contains invalid contents."""
  pass


class EmptyCorpusException(UserError):
  """Raised in case a corpus contains no data."""
  pass


class LlvmException(CLgenError):
  """LLVM Error."""
  pass


class OptException(LlvmException):
  """LLVM opt error."""
  pass


class BadCodeException(CLgenError):
  """Code is bad."""
  pass


class ClangException(BadCodeException):
  """An error from clang."""
  pass


class ClangTimeout(ClangException):
  """Clang failed to terminate without time limit."""
  pass


class ClangFormatException(ClangException):
  """An error from clang-format."""
  pass


class UglyCodeException(CLgenError):
  """Code is ugly."""
  pass


class InstructionCountException(UglyCodeException):
  """Instruction count error."""
  pass


class NoCodeException(UglyCodeException):
  """Sample contains no code."""
  pass


class RewriterException(UglyCodeException):
  """Program rewriter error."""
  pass


class GPUVerifyException(UglyCodeException):
  """GPUVerify found a bug."""
  pass


class GPUVerifyTimeoutException(GPUVerifyException):
  """GPUVerify timed out."""
  pass


class FeaturesError(CLgenError):
  """Thrown in case of error during features encoding."""
  pass


class VocabError(CLgenError):
  """A character sequence is not in the atomizer's vocab"""
  pass


class InvalidVocab(VocabError):
  """An invalid atomizer vocabulary"""
  pass
