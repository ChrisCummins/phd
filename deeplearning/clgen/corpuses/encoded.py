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
"""This file defines a database for encoded content files."""
import datetime
import multiprocessing
import pickle
import time
import typing

import numpy as np
import progressbar
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.sql import func

from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from labm8.py import app
from labm8.py import humanize
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  """Meta table for encoded content files database."""

  __tablename__ = "meta"

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class EncodedContentFile(Base):
  """A single encoded content file."""

  __tablename__ = "encoded_contentfiles"

  # The ID of the PreprocessedContentFile.
  id: int = sql.Column(sql.Integer, primary_key=True)
  # We store the vocabulary indices array as a string of period-separated
  # integers, e.g. '0.1.2.0.1'. To access the values as an array of integers,
  # use EncodedContentFile.indices_array.
  data: str = sql.Column(
    sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False
  )
  tokencount: int = sql.Column(sql.Integer, nullable=False)
  # The number of milliseconds encoding took.
  encoding_time_ms: int = sql.Column(sql.Integer, nullable=False)
  # Encoding is parallelizable, so the actual wall time of encoding may be much
  # less than the sum of all encoding_time_ms. This column counts the effective
  # number of "real" milliseconds during encoding between the last encoded
  # result and this result coming in. The idea is that summing this column
  # provides an accurate total of the actual time spent encoding an entire
  # corpus. Will be <= encoding_time_ms.
  wall_time_ms: int = sql.Column(sql.Integer, nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @staticmethod
  def DataStringToNumpyArray(data: str) -> np.ndarray:
    """Convert the 'data' string to a numpy array."""
    return np.array([int(x) for x in data.split(".")], dtype=np.int32)

  @staticmethod
  def NumpyArrayToDataString(array: np.ndarray) -> str:
    """Convert the 'data' string to a numpy array."""
    return ".".join(str(x) for x in array)

  @property
  def indices_array(self) -> np.ndarray:
    """The numpy array of the encoded data."""
    return self.DataStringToNumpyArray(self.data)

  @classmethod
  def FromPreprocessed(
    cls,
    preprocessed_cf: preprocessed.PreprocessedContentFile,
    atomizer: atomizers.AtomizerBase,
    eof: str,
  ) -> "EncodedContentFile":
    """Instantiate an EncodedContentFile from a preprocessed file.

    Args:
      preprocessed_cf: A PreprocessedContentFile instance.
      atomizer: The atomizer to encode using.
      eof: An end-of-file marker which is concatenated to the encoded sequence.

    Returns:
      An EncodedContentFile instance.
    """
    start_time = time.time()
    data = atomizer.AtomizeString(preprocessed_cf.text)
    encoding_time_ms = int((time.time() - start_time) * 1000)
    return EncodedContentFile(
      id=preprocessed_cf.id,
      # Encode the end-of-file marker separately to ensure that it resolves to
      # the correct token. For example if the vocabulary contains 'a', 'b',
      # and 'ab', then a content file 'a' with EOF marker 'b' would be encoded
      # as 'ab', instead of 'a'+'b'.
      data=cls.NumpyArrayToDataString(
        np.concatenate((data, atomizer.AtomizeString(eof)))
      ),
      tokencount=len(data),
      encoding_time_ms=encoding_time_ms,
      wall_time_ms=encoding_time_ms,  # The outer-loop may change this.
      date_added=datetime.datetime.utcnow(),
    )


def EncoderWorker(
  job: internal_pb2.EncoderWorker,
) -> typing.Optional[EncodedContentFile]:
  """Encode a single content file."""
  # TODO(cec): There is a bug in the atomizer creation logic such that the
  # derived atomizer is not always capable of encoding the preprocessed files.
  # Once this has been fixed, there is no need to catch the VocabError here,
  # and EncoderWorker can always return an EncodedContentFile instance.
  try:
    return EncodedContentFile.FromPreprocessed(
      preprocessed.PreprocessedContentFile(id=job.id, text=job.text),
      pickle.loads(job.pickled_atomizer),
      job.contentfile_separator,
    )
  except errors.VocabError:
    return None


class EncodedContentFiles(sqlutil.Database):
  """A database of encoded pre-processed contentfiles."""

  def __init__(self, url: str, must_exist: bool = False):
    super(EncodedContentFiles, self).__init__(url, Base, must_exist=must_exist)

  def Create(
    self,
    p: preprocessed.PreprocessedContentFiles,
    atomizer: atomizers.AtomizerBase,
    contentfile_separator: str,
  ) -> bool:
    """Populate the encoded contentfiles database.

    Args:
      p: A PreprocessedContentFiles database.
      atomizer: An AtomizerBase instance.
      contentfile_separator: The contentfile separator.

    Returns:
      True if work was done, else False.

    Raises:
      EmptyCorpusException: If the PreprocessedContentFiles database has
        no files.
    """
    with self.Session() as session:
      if not self.IsDone(session):
        self.Import(session, p, atomizer, contentfile_separator)
        self.SetDone(session)
        session.commit()

      # Logging output.
      num_files = session.query(EncodedContentFile).count()
      token_count, total_walltime, total_time, = session.query(
        func.sum(EncodedContentFile.tokencount),
        func.sum(EncodedContentFile.wall_time_ms),
        func.sum(EncodedContentFile.encoding_time_ms),
      ).first()
    app.Log(
      1,
      "Encoded %s files in %s ms (%.2fx speedup).",
      humanize.Commas(num_files),
      humanize.Commas(total_walltime),
      total_time / total_walltime,
    )
    app.Log(
      1,
      "Encoded corpus: %s tokens, %s files.",
      humanize.Commas(token_count),
      humanize.Commas(num_files),
    )

  @property
  def size(self):
    """Return the total number of files in the encoded corpus."""
    with self.Session() as session:
      return session.query(EncodedContentFile).count()

  @property
  def token_count(self) -> int:
    """Return the total number of tokens in the encoded corpus.

    This excludes the EOF markers which are appended to each encoded text.
    """
    with self.Session() as session:
      return session.query(func.sum(EncodedContentFile.tokencount)).scalar()

  def IsDone(self, session: sqlutil.Session):
    if session.query(Meta).filter(Meta.key == "done").first():
      return True
    else:
      return False

  def SetDone(self, session: sqlutil.Session):
    session.add(Meta(key="done", value="yes"))

  def Import(
    self,
    session: sqlutil.Session,
    preprocessed_db: preprocessed.PreprocessedContentFiles,
    atomizer: atomizers.AtomizerBase,
    contentfile_separator: str,
  ) -> None:
    with preprocessed_db.Session() as p_session:
      query = p_session.query(preprocessed.PreprocessedContentFile).filter(
        preprocessed.PreprocessedContentFile.preprocessing_succeeded == True,
        ~preprocessed.PreprocessedContentFile.id.in_(
          session.query(EncodedContentFile.id).all()
        ),
      )
      jobs = [
        internal_pb2.EncoderWorker(
          id=x.id,
          text=x.text,
          contentfile_separator=contentfile_separator,
          pickled_atomizer=pickle.dumps(atomizer),
        )
        for x in query
      ]
      if not jobs:
        raise errors.EmptyCorpusException(
          "Pre-processed corpus contains no files: " f"'{preprocessed_db.url}'"
        )

      app.Log(
        1,
        "Encoding %s of %s preprocessed files",
        humanize.Commas(query.count()),
        humanize.Commas(
          p_session.query(preprocessed.PreprocessedContentFile)
          .filter(
            preprocessed.PreprocessedContentFile.preprocessing_succeeded == True
          )
          .count()
        ),
      )
      pool = multiprocessing.Pool()
      bar = progressbar.ProgressBar(max_value=len(jobs))
      last_commit = time.time()
      wall_time_start = time.time()
      for encoded_cf in bar(pool.imap_unordered(EncoderWorker, jobs)):
        wall_time_end = time.time()
        # TODO(cec): Remove the if check once EncoderWorker no longer returns
        # None on atomizer encode error.
        if encoded_cf:
          encoded_cf.wall_time_ms = int(
            (wall_time_end - wall_time_start) * 1000
          )
          session.add(encoded_cf)
        wall_time_start = wall_time_end
        if wall_time_end - last_commit > 10:
          session.commit()
          last_commit = wall_time_end

  @staticmethod
  def GetVocabFromMetaTable(session) -> typing.Dict[str, int]:
    """Read a vocabulary dictionary from the 'Meta' table of a database."""
    q = session.query(Meta.value).filter(Meta.key == "vocab_size")
    if not q.first():
      return {}

    vocab_size = int(q.one()[0])
    q = session.query(Meta.value)
    return {
      q.filter(Meta.key == f"vocab_{i}").one()[0]: i for i in range(vocab_size)
    }

  @staticmethod
  def StoreVocabInMetaTable(
    session: sqlutil.Session, vocabulary: typing.Dict[str, int]
  ) -> None:
    """Store a vocabulary dictionary in the 'Meta' table of a database."""
    q = session.query(encoded.Meta).filter(encoded.Meta.key.like("vocab_%"))
    q.delete(synchronize_session=False)

    session.add(encoded.Meta(key="vocab_size", value=str(len(vocabulary))))
    session.add_all(
      [encoded.Meta(key=f"vocab_{v}", value=k) for k, v in vocabulary.items()]
    )
