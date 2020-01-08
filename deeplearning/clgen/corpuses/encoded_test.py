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
"""Unit tests for //deeplearning/clgen/corpuses:encoded.py."""
import pathlib
import tempfile

import numpy as np

from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

pytest_plugins = ["deeplearning.clgen.tests.fixtures"]


@test.Fixture(scope="function")
def abc_atomizer() -> atomizers.AsciiCharacterAtomizer:
  """A test fixture which returns a simply atomizer."""
  return atomizers.AsciiCharacterAtomizer.FromText("abcde")


@test.Fixture(scope="function")
def abc_preprocessed() -> preprocessed.PreprocessedContentFile:
  """A test fixture which returns a preprocessed content file."""
  return preprocessed.PreprocessedContentFile(id=123, text="aabbccddee")


@test.Fixture(scope="function")
def temp_db() -> encoded.EncodedContentFiles:
  """A test fixture which returns an empty EncodedContentFiles db."""
  with tempfile.TemporaryDirectory() as d:
    yield encoded.EncodedContentFiles(f"sqlite:///{d}/test.db")


# EncodedContentFile.FromPreprocessed() tests.


def test_EncodedContentFile_FromPreprocessed_id(abc_atomizer, abc_preprocessed):
  """Test that id is the same as the preprocessed content file."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, eof="a"
  )
  assert enc.id == abc_preprocessed.id


def test_EncodedContentFile_FromPreprocessed_indices_array(
  abc_atomizer, abc_preprocessed
):
  """Test that sha256 is the same as the preprocessed content file."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, eof="a"
  )
  np.testing.assert_array_equal(
    np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0], dtype=np.int32),
    enc.indices_array,
  )


def test_EncodedContentFile_FromPreprocessed_tokencount(
  abc_atomizer, abc_preprocessed
):
  """Test that tokencount is the length of the array (minus EOF marker)."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, eof="a"
  )
  # Single character encoding, so tokencount is the length of the string.
  assert len("aabbccddee") == enc.tokencount


def test_EncodedContentFile_FromPreprocessed_encoding_time_ms(
  abc_atomizer, abc_preprocessed
):
  """Test that encoding time is set."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, eof="a"
  )
  assert enc.encoding_time_ms is not None


def test_EncodedContentFile_FromPreprocessed_date_added(
  abc_atomizer, abc_preprocessed
):
  """Test that date_added is set."""
  enc = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, eof="a"
  )
  assert enc.date_added


# EncodedContentFiles tests.


def test_EncodedContentFiles_indices_array_equivalence(
  temp_db: encoded.EncodedContentFiles,
  abc_preprocessed: preprocessed.PreprocessedContentFile,
  abc_atomizer: atomizers.AsciiCharacterAtomizer,
):
  """Test that indices_array is equivalent across db sessions."""
  # Session 1: Add encoded file.
  enc = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, "a"
  )
  with temp_db.Session(commit=True) as session:
    array_in = enc.indices_array
    session.add(enc)

  # Session 2: Get the encoded file.
  with temp_db.Session() as session:
    enc = session.query(encoded.EncodedContentFile).first()
    array_out = enc.indices_array

  np.testing.assert_array_equal(
    np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0], dtype=np.int32), array_in
  )
  np.testing.assert_array_equal(array_in, array_out)


def test_EncodedContentFiles_token_count(
  temp_db: encoded.EncodedContentFiles,
  abc_preprocessed: preprocessed.PreprocessedContentFile,
  abc_atomizer: atomizers.AsciiCharacterAtomizer,
):
  """Test that token_count property returns sum of token_count column."""
  enc1 = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, "a"
  )
  abc_preprocessed.id += 1
  enc2 = encoded.EncodedContentFile.FromPreprocessed(
    abc_preprocessed, abc_atomizer, "a"
  )
  with temp_db.Session(commit=True) as session:
    session.add(enc1)
    session.add(enc2)
    assert 2 == session.query(encoded.EncodedContentFile).count()
  assert 20 == temp_db.token_count


def test_EncodedContentFiles_empty_preprocessed_db(
  temp_db: encoded.EncodedContentFiles,
  abc_atomizer: atomizers.AsciiCharacterAtomizer,
):
  """Test that EmptyCorpusException raised if preprocessed db is empty."""
  with tempfile.TemporaryDirectory() as d:
    p = preprocessed.PreprocessedContentFiles(
      f"sqlite:///{pathlib.Path(d)}/preprocessed.db"
    )
    with test.Raises(errors.EmptyCorpusException):
      temp_db.Create(p, abc_atomizer, "\n\n")


if __name__ == "__main__":
  test.Main()
