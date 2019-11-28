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
import contextlib
import datetime
import hashlib
import multiprocessing
import os
import pathlib
import subprocess
import tempfile
import time
import typing

import progressbar
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.sql import func

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import internal_pb2
from labm8.py import app
from labm8.py import fs
from labm8.py import humanize
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = 'meta'

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class PreprocessedContentFile(Base):
  __tablename__ = 'preprocessed_contentfiles'

  id: int = sql.Column(sql.Integer, primary_key=True)
  # Relative path of the input file within the content files.
  input_relpath: str = sql.Column(sql.String(3072), nullable=False, unique=True)
  # Checksum of the input file.
  input_sha256: str = sql.Column(sql.String(64), nullable=False)
  input_charcount = sql.Column(sql.Integer, nullable=False)
  input_linecount = sql.Column(sql.Integer, nullable=False)
  # Checksum of the preprocessed file.
  sha256: str = sql.Column(sql.String(64), nullable=False, index=True)
  charcount = sql.Column(sql.Integer, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  text: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                         nullable=False)
  # True if pre-processing succeeded, else False.
  preprocessing_succeeded: bool = sql.Column(sql.Boolean, nullable=False)
  # The number of milliseconds pre-preprocessing took.
  preprocess_time_ms: int = sql.Column(sql.Integer, nullable=False)
  # Pre-processing is parallelizable, so the actual wall time of pre-processing
  # may be much less than the sum of all preprocess_time_ms. This column
  # counts the effective number of "real" milliseconds during pre-processing
  # between the last pre-processed result and this result coming in. The idea
  # is that summing this column provides an accurate total of the actual time
  # spent pre-processing an entire corpus. Will be <= preprocess_time_ms.
  wall_time_ms: int = sql.Column(sql.Integer, nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime,
                                             nullable=False,
                                             default=datetime.datetime.utcnow)

  @classmethod
  def FromContentFile(
      cls, contentfile_root: pathlib.Path, relpath: pathlib.Path,
      preprocessors_: typing.List[str]) -> 'PreprocessedContentFile':
    """Instantiate a PreprocessedContentFile."""
    start_time = time.time()
    input_text = ''
    preprocessing_succeeded = False
    try:
      with open(contentfile_root / relpath) as f:
        input_text = f.read()
      text = preprocessors.Preprocess(input_text, preprocessors_)
      preprocessing_succeeded = True
    except UnicodeDecodeError as e:
      text = 'Unicode error'
    except ValueError as e:
      # BadCodeException subclasses ValueError. Catch the more general
      # ValueError so that custom preprocessors can raise ValueError and don't
      # have to depend on CLgen sources.
      text = str(e)
    end_time = time.time()
    preprocess_time_ms = int((end_time - start_time) * 1000)
    input_text_stripped = input_text.strip()
    return cls(
        input_relpath=relpath,
        input_sha256=GetFileSha256(contentfile_root / relpath),
        input_charcount=len(input_text_stripped),
        input_linecount=len(input_text_stripped.split('\n')),
        sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        charcount=len(text),
        linecount=len(text.split('\n')),
        text=text,
        preprocessing_succeeded=preprocessing_succeeded,
        preprocess_time_ms=preprocess_time_ms,
        wall_time_ms=preprocess_time_ms,  # The outer-loop may change this.
        date_added=datetime.datetime.utcnow(),
    )


def PreprocessorWorker(
    job: internal_pb2.PreprocessorWorker) -> PreprocessedContentFile:
  """The inner loop of a parallelizable pre-processing job."""
  return PreprocessedContentFile.FromContentFile(
      pathlib.Path(job.contentfile_root), job.relpath, job.preprocessors)


class PreprocessedContentFiles(sqlutil.Database):
  """A database of pre-processed contentfiles."""

  def __init__(self, url: str, must_exist: bool = False):
    super(PreprocessedContentFiles, self).__init__(url,
                                                   Base,
                                                   must_exist=must_exist)

  def Create(self, config: corpus_pb2.Corpus):
    with self.Session() as session:
      if not self.IsDone(session):
        self.Import(session, config)
        self.SetDone(session)
        session.commit()

      # Logging output.
      num_input_files = session.query(PreprocessedContentFile).count()
      num_files = session.query(PreprocessedContentFile).filter(
          PreprocessedContentFile.preprocessing_succeeded == True).count()
      input_chars, input_lines, total_walltime, total_time, = session.query(
          func.sum(PreprocessedContentFile.charcount),
          func.sum(PreprocessedContentFile.linecount),
          func.sum(PreprocessedContentFile.wall_time_ms),
          func.sum(PreprocessedContentFile.preprocess_time_ms),
      ).first()
      char_count, line_count = session.query(
          func.sum(PreprocessedContentFile.charcount),
          func.sum(PreprocessedContentFile.linecount),
      ).filter(PreprocessedContentFile.preprocessing_succeeded == True).first()
    app.Log(1, 'Content files: %s chars, %s lines, %s files.',
            humanize.Commas(input_chars), humanize.Commas(input_lines),
            humanize.Commas(num_input_files))
    app.Log(1, 'Pre-processed %s files in %s (%.2fx speedup).',
            humanize.Commas(num_input_files),
            humanize.Duration((total_walltime or 0) / 1000),
            (total_time or 1) / (total_walltime or 1))
    app.Log(1, 'Pre-processing discard rate: %.1f%% (%s files).',
            (1 - (num_files / max(num_input_files, 1))) * 100,
            humanize.Commas(num_input_files - num_files))
    app.Log(1, 'Pre-processed corpus: %s chars, %s lines, %s files.',
            humanize.Commas(char_count), humanize.Commas(line_count),
            humanize.Commas(num_files))

  def IsDone(self, session: sqlutil.Session):
    if session.query(Meta).filter(Meta.key == 'done').first():
      return True
    else:
      return False

  def SetDone(self, session: sqlutil.Session):
    session.add(Meta(key='done', value='yes'))

  def Import(self, session: sqlutil.Session, config: corpus_pb2.Corpus) -> None:
    with self.GetContentFileRoot(config) as contentfile_root:
      relpaths = set(self.GetImportRelpaths(contentfile_root))
      done = set(
          [x[0] for x in session.query(PreprocessedContentFile.input_relpath)])
      todo = relpaths - done
      app.Log(1, 'Preprocessing %s of %s content files',
              humanize.Commas(len(todo)), humanize.Commas(len(relpaths)))
      jobs = [
          internal_pb2.PreprocessorWorker(
              contentfile_root=str(contentfile_root),
              relpath=t,
              preprocessors=config.preprocessor) for t in todo
      ]
      pool = multiprocessing.Pool()
      bar = progressbar.ProgressBar(max_value=len(jobs))
      last_commit = time.time()
      wall_time_start = time.time()
      for preprocessed_cf in bar(pool.imap_unordered(PreprocessorWorker, jobs)):
        wall_time_end = time.time()
        preprocessed_cf.wall_time_ms = (int(
            (wall_time_end - wall_time_start) * 1000))
        wall_time_start = wall_time_end
        session.add(preprocessed_cf)
        if wall_time_end - last_commit > 10:
          session.commit()
          last_commit = wall_time_end

  @contextlib.contextmanager
  def GetContentFileRoot(self, config: corpus_pb2.Corpus) -> pathlib.Path:
    """Get the path of the directory containing content files.

    If the corpus is a local directory, this simply returns the path. Otherwise,
    this method creates a temporary copy of the files which can be used within
    the scope of this context.

    Args:
      config: The corpus config proto.

    Returns:
      The path of a directory containing content files.
    """
    if config.HasField('local_directory'):
      yield pathlib.Path(ExpandConfigPath(config.local_directory))
    elif config.HasField('local_tar_archive'):
      with tempfile.TemporaryDirectory(prefix='clgen_corpus_') as d:
        start_time = time.time()
        cmd = [
            'tar', '-xf',
            str(ExpandConfigPath(config.local_tar_archive)), '-C', d
        ]
        subprocess.check_call(cmd)
        app.Log(1, 'Unpacked %s in %s ms',
                ExpandConfigPath(config.local_tar_archive).name,
                humanize.Commas(int((time.time() - start_time) * 1000)))
        yield pathlib.Path(d)
    else:
      raise NotImplementedError

  @property
  def size(self) -> int:
    """Return the total number of files in the pre-processed corpus.

    This excludes contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return session.query(PreprocessedContentFile).filter(
          PreprocessedContentFile.preprocessing_succeeded == True).count()

  @property
  def input_size(self) -> int:
    """Return the total number of files in the pre-processed corpus.

    This *includes* contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return session.query(PreprocessedContentFile).count()

  @property
  def char_count(self) -> int:
    """Get the total number of characters in the pre-processed corpus.

    This excludes contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return session.query(func.sum(PreprocessedContentFile.charcount)).filter(
          PreprocessedContentFile.preprocessing_succeeded == True).scalar()

  @property
  def line_count(self) -> int:
    """Get the total number of lines in the pre-processed corpus.

    This excludes contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return session.query(func.sum(PreprocessedContentFile.linecount)).filter(
          PreprocessedContentFile.preprocessing_succeeded == True).scalar()

  @property
  def input_char_count(self) -> int:
    """Get the total number of characters in the input content files."""
    with self.Session() as session:
      return session.query(func.sum(
          PreprocessedContentFile.input_charcount)).scalar()

  @property
  def input_line_count(self) -> int:
    """Get the total number of characters in the input content files."""
    with self.Session() as session:
      return session.query(func.sum(
          PreprocessedContentFile.input_linecount)).scalar()

  def GetImportRelpaths(self,
                        contentfile_root: pathlib.Path) -> typing.List[str]:
    """Get relative paths to all files in the content files directory.

    Args:
      contentfile_root: The root of the content files directory.

    Returns:
      A list of paths relative to the content files root.

    Raises:
      EmptyCorpusException: If the content files directory is empty.
    """
    with fs.chdir(contentfile_root):
      find_output = subprocess.check_output(['find', '.', '-type',
                                             'f']).decode('utf-8').strip()
      if not find_output:
        raise errors.EmptyCorpusException(
            f"Empty content files directory: '{contentfile_root}'")
      return find_output.split('\n')


def ExpandConfigPath(path: str) -> pathlib.Path:
  return pathlib.Path(os.path.expandvars(path)).expanduser().absolute()


def GetFileSha256(path: pathlib.Path) -> str:
  with open(path, 'rb') as f:
    return hashlib.sha256(f.read()).hexdigest()
