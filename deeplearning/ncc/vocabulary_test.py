"""Unit tests for //deeplearning/ncc:vocabulary."""

import pytest
from absl import flags

from deeplearning.ncc import vocabulary
from labm8 import bazelutil
from labm8 import test


FLAGS = flags.FLAGS

VOCABULARY_PATH = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/vocabulary.zip')


@pytest.fixture(scope='session')
def vocab() -> vocabulary.VocabularyZipFile:
  """Test fixture which yields a vocabulary zip file instance as a ctx mngr."""
  with vocabulary.VocabularyZipFile(VOCABULARY_PATH) as v:
    yield v


def test_VocabularyZipFile_dictionary_type(
    vocab: vocabulary.VocabularyZipFile):
  """Test that dictionary is a dict."""
  assert isinstance(vocab.dictionary, dict)


def test_VocabularyZipFile_dictionary_size(
    vocab: vocabulary.VocabularyZipFile):
  """Test that dictionary contains at least 2 values (1 for !UNK, +1 other)."""
  assert len(vocab.dictionary) >= 2


def test_VocabularyZipFile_dictionary_values_are_unique(
    vocab: vocabulary.VocabularyZipFile):
  """Test that values in vocabulary are unique."""
  assert len(set(vocab.dictionary.values())) == len(vocab.dictionary.values())


def test_VocabularyZipFile_dictionary_values_are_positive_integers(
    vocab: vocabulary.VocabularyZipFile):
  """Test that values in vocabulary are unique."""
  for value in vocab.dictionary.values():
    assert value >= 0


def test_VocabularyZipFile_cutoff_stmts_type(
    vocab: vocabulary.VocabularyZipFile):
  """Test that cutoff_stmts is a set."""
  assert isinstance(vocab.cutoff_stmts, set)


def test_VocabularyZipFile_unknown_token_index_type(
    vocab: vocabulary.VocabularyZipFile):
  """Test that unknown token index is an integer."""
  assert isinstance(vocab.unknown_token_index, int)
  assert vocab.unknown_token_index > 0


def test_VocabularyZipFile_unknown_token_index_value(
    vocab: vocabulary.VocabularyZipFile):
  """Test that unknown token index is positive."""
  assert vocab.unknown_token_index > 0


if __name__ == '__main__':
  test.Main()
