"""Unit tests for ///cxx_test."""
import sys

import numpy as np
import pytest
from absl import app
from absl import flags

from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def abc_atomizer():
  return atomizers.AsciiCharacterAtomizer.FromText('abcde')


@pytest.fixture(scope='function')
def abc_preprocessed():
  return preprocessed.PreprocessedContentFile(id=123, text='aabbccddee')


# EncodedContentFile.FromPreprocessed() tests.

def test_EncodedContentFile_FromPreprocessed_id(
    abc_atomizer, abc_preprocessed):
  """Test that id is the same as the preprocessed content file."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, eof='a')
  assert enc.id == abc_preprocessed.id


def test_EncodedContentFile_FromPreprocessed_data(
    abc_atomizer, abc_preprocessed):
  """Test that sha256 is the same as the preprocessed content file."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, eof='a')
  assert np.array_equal(np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0]),
                        enc.data)


def test_EncodedContentFile_FromPreprocessed_encoding_time_ms(
    abc_atomizer, abc_preprocessed):
  """Test that encoding time is set."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, eof='a')
  assert enc.encoding_time_ms is not None


def test_EncodedContentFile_FromPreprocessed_date_added(
    abc_atomizer, abc_preprocessed):
  """Test that date_added is set."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, eof='a')
  assert enc.date_added


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
