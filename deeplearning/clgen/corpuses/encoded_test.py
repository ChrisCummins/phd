"""Unit tests for ///cxx_test."""
import pathlib
import sys
import tempfile

import numpy as np
import pytest
from absl import app
from absl import flags

from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def abc_atomizer() -> atomizers.AsciiCharacterAtomizer:
  """A test fixture which returns a simply atomizer."""
  return atomizers.AsciiCharacterAtomizer.FromText('abcde')


@pytest.fixture(scope='function')
def abc_preprocessed() -> preprocessed.PreprocessedContentFile:
  """A test fixture which returns a preprocessed content file."""
  return preprocessed.PreprocessedContentFile(id=123, text='aabbccddee')


@pytest.fixture(scope='function')
def temp_db() -> encoded.EncodedContentFiles:
  """A test fixture which returns an empty EncodedContentFiles db."""
  with tempfile.TemporaryDirectory() as d:
    yield encoded.EncodedContentFiles(pathlib.Path(d) / 'test.db')


# EncodedContentFile.FromPreprocessed() tests.

def test_EncodedContentFile_FromPreprocessed_id(
    abc_atomizer, abc_preprocessed):
  """Test that id is the same as the preprocessed content file."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, eof='a')
  assert enc.id == abc_preprocessed.id


def test_EncodedContentFile_FromPreprocessed_indices_array(
    abc_atomizer, abc_preprocessed):
  """Test that sha256 is the same as the preprocessed content file."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, eof='a')
  np.testing.assert_array_equal(
      np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0], dtype=np.int32),
      enc.indices_array)


def test_EncodedContentFile_FromPreprocessed_tokencount(
    abc_atomizer, abc_preprocessed):
  """Test that tokencount is the length of the array (minus EOF marker)."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, eof='a')
  # Single character encoding, so tokencount is the length of the string.
  assert len('aabbccddee') == enc.tokencount


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


# EncodedContentFiles tests.

def test_EncodedContentFiles_indices_array_equivalence(
    temp_db: encoded.EncodedContentFiles,
    abc_preprocessed: preprocessed.PreprocessedContentFile,
    abc_atomizer: atomizers.AsciiCharacterAtomizer):
  """Test that indices_array is equivalent across db sessions."""
  # Session 1: Add encoded file.
  enc = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, 'a')
  with temp_db.Session(commit=True) as session:
    array_in = enc.indices_array
    session.add(enc)

  # Session 2: Get the encoded file.
  with temp_db.Session() as session:
    enc = session.query(encoded.EncodedContentFile).first()
    array_out = enc.indices_array

  np.testing.assert_array_equal(
      np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0], dtype=np.int32), array_in)
  np.testing.assert_array_equal(array_in, array_out)


def test_EncodedContentFiles_token_count(
    temp_db: encoded.EncodedContentFiles,
    abc_preprocessed: preprocessed.PreprocessedContentFile,
    abc_atomizer: atomizers.AsciiCharacterAtomizer):
  """Test that token_count property returns sum of token_count column."""
  enc1 = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, 'a')
  abc_preprocessed.id += 1
  enc2 = encoded.EncodedContentFile.FromPreprocessed(
      abc_preprocessed, abc_atomizer, 'a')
  with temp_db.Session(commit=True) as session:
    session.add(enc1)
    session.add(enc2)
    assert 2 == session.query(encoded.EncodedContentFile).count()
  assert 20 == temp_db.token_count


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
