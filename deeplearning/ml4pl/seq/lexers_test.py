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
"""Unit tests for //deeplearning/ml4pl/seq:lexer."""
import random
import string
from typing import Dict

import numpy as np

from deeplearning.ml4pl.seq import lexers
from labm8.py import decorators
from labm8.py import test


FLAGS = test.FLAGS


@test.Fixture(scope="function", params=list(lexers.LexerType))
def lexer_type(request) -> lexers.LexerType:
  """Test fixture for lexer types."""
  return request.param


@test.Fixture(scope="function", params=({"abc": 0, "bcd": 1},))
def vocabulary(request) -> Dict[str, int]:
  """Test fixture for initial vocabs."""
  return request.param


@test.Fixture(scope="function", params=(10, 1024, 1024 * 1024))
def max_encoded_length(request) -> int:
  """Test fixture for lexer max encoded lengths."""
  return request.param


@test.Fixture(scope="function")
def lexer(
  lexer_type: lexers.LexerType,
  vocabulary: Dict[str, int],
  max_encoded_length: int,
) -> lexers.Lexer:
  """A test fixture which returns a lexer."""
  return lexers.Lexer(
    type=lexer_type,
    vocabulary=vocabulary,
    max_encoded_length=max_encoded_length,
  )


def CreateRandomString(min_length: int = 1, max_length: int = 1024) -> str:
  """Generate a random string."""
  return "".join(
    random.choice(string.ascii_lowercase)
    for _ in range(random.randint(min_length, max_length))
  )


@decorators.loop_for(seconds=30)
def test_fuzz_Lex(lexer: lexers.Lexer, max_encoded_length: int):
  """Fuzz the lexer."""
  texts_count = random.randint(1, 128)
  texts = [CreateRandomString() for _ in range(texts_count)]

  lexed = lexer.Lex(texts)
  assert len(lexed) == texts_count
  for encoded in lexed:
    assert len(encoded) <= max_encoded_length
    assert not np.where(encoded > lexer.vocabulary_size)[0].size


if __name__ == "__main__":
  test.Main()
