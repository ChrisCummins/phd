# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Custom exception hierarchy used by CLgen."""


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


class BadCodeException(ValueError):
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


class RewriterException(ClangException):
  """Program rewriter error."""
  pass


class InstructionCountException(BadCodeException):
  """Instruction count error."""
  pass


class NoCodeException(BadCodeException):
  """Sample contains no code."""
  pass


class GPUVerifyException(BadCodeException):
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


class InvalidStartText(InvalidVocab):
  """A Sampler's start text cannot be encoded using the corpus vocabulary."""
  pass


class InvalidSymtokTokens(InvalidVocab):
  """A Sampler's symmetrical depth tokens cannot be encoded."""
  pass
