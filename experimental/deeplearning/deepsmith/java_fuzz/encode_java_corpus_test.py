"""Unit tests for //experimental/deeplearning/deepsmith/java_fuzz:encode_java_corpus."""
import pathlib

import pytest

from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed
from experimental.deeplearning.deepsmith.java_fuzz import encode_java_corpus
from labm8.py import test

FLAGS = test.FLAGS


@pytest.fixture(scope="function")
def preprocessed_db(
  tempdir: pathlib.Path,
) -> preprocessed.PreprocessedContentFile:
  db = preprocessed.PreprocessedContentFiles(
    f"sqlite:///{tempdir}/preprocessed"
  )
  return db


@pytest.fixture(scope="function")
def encoded_db(tempdir: pathlib.Path) -> encoded.EncodedContentFiles:
  db = encoded.EncodedContentFiles(f"sqlite:///{tempdir}/encoded")
  return db


def test_EmbedVocabInMetaTable(encoded_db: encoded.EncodedContentFiles):
  """Test that meta table stores vocabulary."""
  with encoded_db.Session() as s:
    encode_java_corpus.EmbedVocabInMetaTable(s, {"a": 0, "b": 1, "c": 2})
    s.flush()
    vocab_size = s.query(encoded.Meta.value).filter(
      encoded.Meta.key == "vocab_size"
    )
    assert vocab_size.one()[0] == "3"

    for i, val in enumerate(["a", "b", "c"]):
      assert (
        s.query(encoded.Meta.value)
        .filter(encoded.Meta.key == f"vocab_{i}")
        .one()[0]
        == val
      )


def test_EmbedVocabInMetaTable_GetVocabFromMetaTable_equivalence(
  encoded_db: encoded.EncodedContentFiles,
):
  """Test store and load to meta table."""
  with encoded_db.Session(commit=True) as s:
    encode_java_corpus.EmbedVocabInMetaTable(s, {"a": 0, "b": 1, "c": 2})
  with encoded_db.Session() as s:
    vocab = encode_java_corpus.GetVocabFromMetaTable(s)
  assert vocab == {"a": 0, "b": 1, "c": 2}


def _PreprocessedContentFile(
  relpath: str, text: str, preprocessing_succeeded: bool
) -> preprocessed.PreprocessedContentFile:
  return preprocessed.PreprocessedContentFile(
    input_relpath=relpath,
    input_sha256="000",
    input_charcount=0,
    input_linecount=0,
    sha256="000",
    charcount=0,
    linecount=0,
    text=text,
    preprocessing_succeeded=preprocessing_succeeded,
    preprocess_time_ms=0,
    wall_time_ms=0,
  )


def _Decode(array, rvocab):
  """Decode an array using the given reverse-lookup vocabulary dictionary."""
  # Dot-separated tokens.
  return ".".join([rvocab[x] for x in array])


def test_EncodeFiles(preprocessed_db, encoded_db):
  with preprocessed_db.Session() as pps:
    pps.add_all(
      [
        _PreprocessedContentFile("a", "abc", True),
        _PreprocessedContentFile("b", "def", False),
        _PreprocessedContentFile("c", "abcghi", True),
      ]
    )
    pps.flush()
    with encoded_db.Session(commit=True) as es:
      assert encode_java_corpus.EncodeFiles(pps, es, 10) == 2

  with encoded_db.Session() as s:
    vocab = encode_java_corpus.GetVocabFromMetaTable(s)
    rvocab = {v: k for k, v in vocab.items()}

    encodeds = [x.indices_array for x in s.query(encoded.EncodedContentFile)]
    decoded = set(_Decode(x, rvocab) for x in encodeds)

  assert decoded == {"a.b.c", "a.b.c.g.h.i"}


if __name__ == "__main__":
  test.Main()
