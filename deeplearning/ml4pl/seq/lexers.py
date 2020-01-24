# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines a lexer interface as described in PACT'17 paper."""
import copy
import enum
import json
from typing import Dict
from typing import List

import numpy as np

from deeplearning.ml4pl.seq import ir2seq_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import progress


FLAGS = app.FLAGS

# The native C++ string encoder binary.
STRING_ENCODER_WORKER = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/seq/string_encoder_worker"
)

TOKEN_LISTS = json.loads(
  bazelutil.DataString("phd/deeplearning/clgen/corpuses/token_lists.json")
)


class LexerType(enum.Enum):
  """The type of lexer."""

  OPENCL = 1
  LLVM = 2


OPENCL_TOKENS = TOKEN_LISTS["opencl"]["tokens"]
LLVM_TOKENS = TOKEN_LISTS["llvm"]["tokens"]


class Lexer(object):
  """A lexer."""

  def __init__(
    self, type: LexerType, vocabulary: Dict[str, int], max_encoded_length: int,
  ):
    self.candidate_tokens = {
      LexerType.LLVM: LLVM_TOKENS,
      LexerType.OPENCL: OPENCL_TOKENS,
    }[type]

    self.vocab = copy.deepcopy(vocabulary)
    self.max_encoded_length = max_encoded_length

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
        string=[text[: self.max_encoded_length] for text in texts],
        vocabulary=self.vocab,
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
