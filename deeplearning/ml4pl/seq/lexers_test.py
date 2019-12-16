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


@test.Fixture(scope="function", params=({}, {"abc": 0, "bcd": 1}))
def initial_vocab(request) -> Dict[str, int]:
  """Test fixture for initial vocabs."""
  return request.param


@test.Fixture(scope="function", params=(10, 1024, 1024 * 1024))
def max_chunk_size(request) -> int:
  """Test fixture for lexer max chunk sizes."""
  return request.param


@test.Fixture(scope="function")
def lexer(
  lexer_type: lexers.LexerType,
  initial_vocab: Dict[str, int],
  max_chunk_size: int,
) -> lexers.Lexer:
  """A test fixture which returns a lexer."""
  return lexers.Lexer(
    type=lexer_type, initial_vocab=initial_vocab, max_chunk_size=max_chunk_size
  )


def CreateRandomString(min_length: int = 1, max_length: int = 1024) -> str:
  """Generate a random string."""
  return "".join(
    random.choice(string.ascii_lowercase)
    for _ in range(random.randint(min_length, max_length))
  )


@decorators.loop_for(seconds=30)
def test_fuzz_Lex(lexer: lexers.Lexer):
  """Fuzz the lexer."""
  texts_count = random.randint(1, 128)
  texts = [CreateRandomString() for _ in range(texts_count)]

  lexed = lexer.Lex(texts)
  assert len(lexed) == texts_count
  for encoded in lexed:
    assert not np.where(encoded > lexer.vocabulary_size)[0].size


if __name__ == "__main__":
  test.Main()
