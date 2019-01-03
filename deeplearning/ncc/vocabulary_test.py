"""Unit tests for //deeplearning/ncc:vocabulary."""
import sys
import typing

import pytest
from absl import app
from absl import flags

from deeplearning.ncc import vocabulary
from labm8 import bazelutil


FLAGS = flags.FLAGS

VOCABULARY_PATH = bazelutil.DataPath(
    'phd/deeplearning/ncc/published_results/vocabulary.zip')


@pytest.fixture(scope='session')
def vocab() -> vocabulary.VocabularyZipFile:
  """Test fixture which yields a vocabulary zip file instance as a ctx mngr."""
  with vocabulary.VocabularyZipFile(VOCABULARY_PATH) as v:
    yield v


def test_VocabularyZipFile_dictionary_pickle_path(
    vocab: vocabulary.VocabularyZipFile):
  """Test that dictionary pickle path is a file."""
  assert vocab.dictionary_pickle.is_file()


def test_VocabularyZipFile_cutoff_stmts_pickle_path(
    vocab: vocabulary.VocabularyZipFile):
  """Test that cutoff_stmts pickle path is a file."""
  assert vocab.cutoff_stmts_pickle.is_file()


def test_VocabularyZipFile_unknown_token_index(
    vocab: vocabulary.VocabularyZipFile):
  """Test that unknown token index is a positive integer."""
  assert isinstance(vocab.unknown_token_index, int)
  assert vocab.unknown_token_index > 0


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
