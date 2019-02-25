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
"""This file defines a database for pre-preprocessed content files."""
import binascii
import datetime
import multiprocessing
import pathlib
import pickle
import time
import typing

import humanize
import numpy as np
import progressbar
import sqlalchemy as sql
from absl import flags
from absl import logging
from sqlalchemy.ext import declarative
from sqlalchemy.sql import func

from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from labm8 import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  """Meta table for encoded content files database."""
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class EncodedContentFile(Base):
  """A single encoded content file."""
  __tablename__ = 'encoded_contentfiles'

  # The ID of the PreprocessedContentFile.
  id: int = sql.Column(sql.Integer, primary_key=True)
  data: bytes = sql.Column(sql.Binary(), nullable=False)
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

  @property
  def indices_array(self) -> np.ndarray:
    """The numpy array of the encoded data."""
    return np.frombuffer(self.data, dtype=np.int32)

  @property
  def sha256_hex(self) -> str:
    """Return the 64 character hexadecimal representation of sha256."""
    return binascii.hexlify(self.sha256).decode('utf-8')

  @classmethod
  def FromPreprocessed(
      cls, preprocessed_cf: preprocessed.PreprocessedContentFile,
      atomizer: atomizers.AtomizerBase, eof: str) -> 'EncodedContentFile':
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
        data=np.concatenate((data, atomizer.AtomizeString(eof))).tostring(),
        tokencount=len(data),
        encoding_time_ms=encoding_time_ms,
        wall_time_ms=encoding_time_ms,  # The outer-loop may change this.
        date_added=datetime.datetime.utcnow())


def EncoderWorker(
    job: internal_pb2.EncoderWorker) -> typing.Optional[EncodedContentFile]:
  """Encode a single content file."""
  # TODO(cec): There is a bug in the atomizer creation logic such that the
  # derived atomizer is not always capable of encoding the preprocessed files.
  # Once this has been fixed, there is no need to catch the VocabError here,
  # and EncoderWorker can always return an EncodedContentFile instance.
  try:
    return EncodedContentFile.FromPreprocessed(
        preprocessed.PreprocessedContentFile(id=job.id, text=job.text),
        pickle.loads(job.pickled_atomizer), job.contentfile_separator)
  except errors.VocabError:
    return None


class EncodedContentFiles(sqlutil.Database):
  """A database of encoded pre-processed contentfiles."""

  def __init__(self, path: pathlib.Path):
    super(EncodedContentFiles, self).__init__(f'sqlite:///{path.absolute()}',
                                              Base)

  def Create(self, p: preprocessed.PreprocessedContentFiles,
             atomizer: atomizers.AtomizerBase,
             contentfile_separator: str) -> bool:
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
    logging.info('Encoded %s files in %s ms (%.2fx speedup).',
                 humanize.intcomma(num_files),
                 humanize.intcomma(total_walltime), total_time / total_walltime)
    logging.info('Encoded corpus: %s tokens, %s files.',
                 humanize.intcomma(token_count), humanize.intcomma(num_files))

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
    if session.query(Meta).filter(Meta.key == 'done').first():
      return True
    else:
      return False

  def SetDone(self, session: sqlutil.Session):
    session.add(Meta(key='done', value='yes'))

  def Import(self, session: sqlutil.Session,
             preprocessed_db: preprocessed.PreprocessedContentFiles,
             atomizer: atomizers.AtomizerBase,
             contentfile_separator: str) -> None:
    with preprocessed_db.Session() as p_session:
      query = p_session.query(preprocessed.PreprocessedContentFile).filter(
          preprocessed.PreprocessedContentFile.preprocessing_succeeded == True,
          ~preprocessed.PreprocessedContentFile.id.in_(
              session.query(EncodedContentFile.id).all()))
      jobs = [
          internal_pb2.EncoderWorker(
              id=x.id,
              text=x.text,
              contentfile_separator=contentfile_separator,
              pickled_atomizer=pickle.dumps(atomizer)) for x in query
      ]
      if not jobs:
        raise errors.EmptyCorpusException(
            "Pre-processed corpus contains no files: "
            f"'{preprocessed_db.url}'")

      logging.info(
          'Encoding %s of %s preprocessed files',
          humanize.intcomma(query.count()),
          humanize.intcomma(
              p_session.query(preprocessed.PreprocessedContentFile).filter(
                  preprocessed.PreprocessedContentFile.preprocessing_succeeded
                  == True).count()))
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
              (wall_time_end - wall_time_start) * 1000)
          session.add(encoded_cf)
        wall_time_start = wall_time_end
        if wall_time_end - last_commit > 10:
          session.commit()
          last_commit = wall_time_end
