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
"""This file defines the logic and training corpuses.

A training corpus is a set of one or more "contentfiles", where each contentfile
is a file containing text to train over.
"""
import os
import pathlib
import subprocess
import tempfile
import time

import checksumdir
import humanize
import numpy as np
from absl import flags
from absl import logging
from sqlalchemy.sql.expression import func

from deeplearning.clgen import cache
from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import corpus_pb2
from labm8 import bazelutil
from labm8 import crypto
from labm8 import hashcache
from labm8 import lockfile
from labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'clgen_local_path_prefix', None,
    'An optional prefix to use when resolving the path to a local directory '
    'or archive. For example, given a corpus which is configured for a '
    'local_directory with value "foo/bar" and a --clgen_local_path_prefix of '
    '"/tmp/", the absolute path of the corpus will resolve to "/tmp/foo/bar". '
    'If the --clgen_local_path_prefix is a directory, the trailing slash must '
    'not be omitted.')


def AssertConfigIsValid(config: corpus_pb2.Corpus) -> corpus_pb2.Corpus:
  """Assert that config proto is valid.

  Args:
    config: A Corpus proto.

  Returns:
    The Corpus proto.

  Raises:
    UserError: If the config is invalid.
  """
  try:
    pbutil.AssertFieldIsSet(config, 'contentfiles')
    pbutil.AssertFieldIsSet(config, 'atomizer')
    pbutil.AssertFieldIsSet(config, 'contentfile_separator')
    # Check that the preprocessor pipeline resolves to preprocessor functions.
    [preprocessors.GetPreprocessorFunction(p) for p in config.preprocessor]

    if config.HasField('greedy_multichar_atomizer'):
      if not config.greedy_multichar_atomizer.tokens:
        raise errors.UserError('GreedyMulticharAtomizer.tokens is empty')
      for atom in config.greedy_multichar_atomizer.tokens:
        if not atom:
          raise errors.UserError(
              'Empty string found in GreedyMulticharAtomizer.tokens is empty')

    return config
  except pbutil.ProtoValueError as e:
    raise errors.UserError(e)


class Corpus(object):
  """Representation of a training corpus.

  Please note corpus instances should be treated as immutable. Upon
  instantiation, a corpus's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: corpus_pb2.Corpus):
    """Instantiate a corpus from a proto config.

    If this is a new corpus, a number of files will be created, which may
    take some time.

    Args:
      config: A Corpus message.

    Raises:
      TypeError: If the config argument is not a Sampler proto.
      UserError: In case the corpus is not found, or config contains invalid
        options.
      EmptyCorpusException: In case the corpus contains no data.
    """
    if not isinstance(config, corpus_pb2.Corpus):
      t = type(config).__name__
      raise TypeError(f"Config must be a Corpus proto. Received: '{t}'")

    # Make a local copy of the configuration.
    self.config = corpus_pb2.Corpus()
    self.config.CopyFrom(AssertConfigIsValid(config))
    self._atomizer = None
    self._created = False

    cache.cachepath('corpus').mkdir(parents=True, exist_ok=True)
    hc = hashcache.HashCache(cache.cachepath('hashcache.db'), 'sha1')
    self.content_id = ResolveContentId(self.config, hc)
    # Database of pre-processed files.
    preprocessed_id = ResolvePreprocessedId(self.content_id, self.config)
    cache.cachepath('corpus', 'preprocessed', preprocessed_id).mkdir(
        exist_ok=True, parents=True)
    preprocessed_db_path = cache.cachepath(
        'corpus', 'preprocessed', preprocessed_id, 'preprocessed.db')
    if (self.config.HasField('content_id') and
        not preprocessed_db_path.is_file()):
      raise errors.UserError(f"Content ID not found: '{self.content_id}'")
    self.preprocessed = preprocessed.PreprocessedContentFiles(
        preprocessed_db_path)
    # Create symlink to contentfiles.
    symlink = pathlib.Path(
        self.preprocessed.url[len('sqlite:///'):]).parent / 'contentfiles'
    if not symlink.is_symlink():
      if config.HasField('local_directory'):
        os.symlink(str(ExpandConfigPath(
            config.local_directory, path_prefix=FLAGS.clgen_local_path_prefix)),
            symlink)
      elif config.HasField('local_tar_archive'):
        os.symlink(str(ExpandConfigPath(
            config.local_tar_archive,
            path_prefix=FLAGS.clgen_local_path_prefix)), symlink)
    # Data of encoded pre-preprocessed files.
    encoded_id = ResolveEncodedId(self.content_id, self.config)
    cache.cachepath('corpus', 'encoded', encoded_id).mkdir(
        exist_ok=True, parents=True)
    self.encoded = encoded.EncodedContentFiles(cache.cachepath(
        'corpus', 'encoded', encoded_id, 'encoded.db'))
    self.atomizer_path = cache.cachepath(
        'corpus', 'encoded', encoded_id, 'atomizer.pkl')
    # Create symlink to preprocessed files.
    symlink = pathlib.Path(
        self.encoded.url[len('sqlite:///'):]).parent / 'preprocessed'
    if not symlink.is_symlink():
      os.symlink(os.path.relpath(
          pathlib.Path(self.preprocessed.url[len('sqlite:///'):]).parent,
          pathlib.Path(self.encoded.url[len('sqlite:///'):]).parent), symlink)
    self.hash = encoded_id
    self.cache = cache.mkcache('corpus', 'encoded', encoded_id)

  def Create(self) -> None:
    """Create the corpus files.

    Raises:
      EmptyCorpusException: If there are no content files, or no successfully
        pre-processed files.
    """
    self._created = True
    logging.info('Content ID: %s', self.content_id)
    preprocessed_lock_path = pathlib.Path(
        self.preprocessed.url[len('sqlite:///'):]).parent / 'LOCK'
    with lockfile.LockFile(preprocessed_lock_path).acquire(
        replace_stale=True, block=True):
      self.preprocessed.Create(self.config)
    if not self.preprocessed.size:
      raise errors.EmptyCorpusException(
          f"Pre-processed corpus contains no files: '{self.preprocessed.url}'")
    encoded_lock_path = pathlib.Path(
        self.encoded.url[len('sqlite:///'):]).parent / 'LOCK'
    with lockfile.LockFile(encoded_lock_path).acquire(
        replace_stale=True, block=True):
      start_time = time.time()
      atomizer = self.atomizer
      logging.info('%s: %s tokens in %s ms', type(atomizer).__name__,
                   humanize.intcomma(atomizer.vocab_size),
                   humanize.intcomma(int((time.time() - start_time) * 1000)))
      self.encoded.Create(self.preprocessed, atomizer,
                          self.config.contentfile_separator)

  @property
  def is_locked(self) -> bool:
    """Return whether the corpus is locked."""
    preprocessed_lock_path = pathlib.Path(
        self.preprocessed.url[len('sqlite:///'):]).parent / 'LOCK'
    if lockfile.LockFile(preprocessed_lock_path).islocked:
      return True
    encoded_lock_path = pathlib.Path(
        self.encoded.url[len('sqlite:///'):]).parent / 'LOCK'
    if lockfile.LockFile(encoded_lock_path).islocked:
      return True
    return False

  def GetTextCorpus(self, shuffle: bool) -> str:
    """Concatenate the entire corpus into a string.

    Args:
      shuffle: If true, randomize order of contentfiles.

    Returns:
      A concatenated corpus string.
    """
    with self.preprocessed.Session() as session:
      query = session.query(preprocessed.PreprocessedContentFile.text).filter(
          preprocessed.PreprocessedContentFile.preprocessing_succeeded == True)
      if shuffle:
        query = query.order_by(func.random())
      return self.config.contentfile_separator.join([x[0] for x in query])

  def GetTrainingData(self, shuffle: bool) -> np.ndarray:
    """Concatenate the entire encoded corpus into an array.

    Args:
      shuffle: If true, randomize order of ebcided contentfiles.

    Returns:
      The encoded corpus.
    """
    # TODO: Can binary numpy strings be concatenated and decoded as one?
    with self.encoded.Session() as session:
      query = session.query(encoded.EncodedContentFile.data)
      if shuffle:
        query = query.order_by(func.random())
      return np.concatenate(
          [np.frombuffer(x[0], dtype=np.int32) for x in query])

  def GetNumContentFiles(self) -> int:
    """Get the number of contentfiles which were pre-processed."""
    with self.preprocessed.Session() as session:
      return session.query(preprocessed.PreprocessedContentFile).count()

  def GetNumPreprocessedFiles(self) -> int:
    """The number of succesfully pre-processed content files."""
    with self.preprocessed.Session() as session:
      return session.query(preprocessed.PreprocessedContentFile.text).filter(
          preprocessed.PreprocessedContentFile.preprocessing_succeeded == True
      ).count()

  @property
  def atomizer(self) -> atomizers.AtomizerBase:
    """Must call Create() first."""
    if not self._created:
      raise ValueError('Must call Create() before accessing atomizer property.')
    if self._atomizer is None:
      if self.atomizer_path.is_file():
        self._atomizer = atomizers.AtomizerBase.FromFile(self.atomizer_path)
      else:
        self._atomizer = self._CreateAtomizer()
    return self._atomizer

  def _CreateAtomizer(self) -> atomizers.AtomizerBase:
    """Creates and caches an atomizer."""
    logging.info('Deriving atomizer from preprocessed corpus')
    corpus_txt = self.GetTextCorpus(shuffle=False)

    if self.config.HasField('ascii_character_atomizer'):
      atomizer = atomizers.AsciiCharacterAtomizer.FromText(corpus_txt)
    elif self.config.HasField('greedy_multichar_atomizer'):
      atoms = set(self.config.greedy_multichar_atomizer.tokens)
      atomizer = atomizers.GreedyAtomizer.FromText(corpus_txt, atoms)
    else:
      raise NotImplementedError

    atomizer.ToFile(self.atomizer_path)
    return atomizer

  @property
  def vocab_size(self) -> int:
    """Get the number of elements in the corpus vocabulary."""
    return self.atomizer.vocab_size

  @property
  def size(self) -> int:
    """Return the size of the atomized corpus."""
    return len(self.GetTrainingData(shuffle=False))

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Corpus):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)


def ExpandConfigPath(path: str,
                     path_prefix: str = None) -> pathlib.Path:
  """Resolve an absolute path from a config proto string field.

  This performs shell-style expansion of $VARS, and prefixes the
  --clgen_local_path_prefix flag value, if it is set.

  Args:
    path: The string value as it appears in the proto.
    path_prefix: An optional string to prepend to the resolved path.

  Returns:
    An absolute path.
  """
  # Set a useful variable for expansion.
  if 'HOME' not in os.environ:
    os.environ['HOME'] = str(pathlib.Path('~').expanduser())
  os.environ['BAZEL_RUNFILES'] = str(bazelutil.DataPath('.'))
  return pathlib.Path(os.path.expandvars(
      (path_prefix or '') + path)).expanduser().absolute()


def ResolveContentId(config: corpus_pb2.Corpus, hc: hashcache.HashCache) -> str:
  """Compute the hash of the input contentfiles.

  This function resolves the unique sha1 checksum of a set of content files.

  Args:
    config: The corpus config proto.
    hc: A hashcache database instance, used for resolving directory hashes.

  Returns:
    A hex encoded sha1 string.
  """
  # We can take a massive shortcut if the content ID is already set in the
  # config proto.
  if config.HasField('content_id'):
    return config.content_id

  start_time = time.time()
  if config.HasField('local_directory'):
    local_directory = ExpandConfigPath(
        config.local_directory, path_prefix=FLAGS.clgen_local_path_prefix)

    # After the first time we compute the hash of a directory, we write it into
    # a file. This is a shortcut to work around the fact that computing the
    # directory checksum is O(n) with respect to the number of files in the
    # directory (even if the directory is already cached by the hash cache).
    # This means that it is the responsibility of the user to delete this cached
    # file if the directory is changed.
    hash_file_path = pathlib.Path(str(local_directory) + '.sha1.txt')
    if hash_file_path.is_file():
      logging.info("Reading directory hash: '%s'.", hash_file_path)
      with open(hash_file_path) as f:
        content_id = f.read().rstrip()
    else:
      # No hash file, so compute the directory hash and create it.
      try:
        content_id = hc.GetHash(local_directory)
      except FileNotFoundError as e:
        raise errors.UserError(e)
      # Create the hash file in the directory so that next time we don't need
      # to reference the hash cache.
      with open(hash_file_path, 'w') as f:
        print(content_id, file=f)
      logging.info("Wrote directory hash: '%s'.", hash_file_path)
  elif config.HasField('local_tar_archive'):
    # This if not an efficient means of getting the hash, as it requires always
    # unpacking the archive and reading the entire contents. It would be nicer
    # to maintain a cache which maps the mtime of tarballs to their content ID,
    # similart to how local_directory is implemented.
    content_id = GetHashOfArchiveContents(
        ExpandConfigPath(config.local_tar_archive,
                         path_prefix=FLAGS.clgen_local_path_prefix))
  else:
    raise NotImplementedError('Unsupported Corpus.contentfiles field value')
  logging.debug('Resolved Content ID %s in %s ms.', content_id,
                humanize.intcomma(int((time.time() - start_time) * 1000)))
  return content_id


def ResolvePreprocessedId(content_id: str, config: corpus_pb2.Corpus) -> str:
  """Compute the hash of a corpus of preprocessed contentfiles.

  The hash is computed from the ID of the input files and the serialized
  representation of the preprocessor pipeline.
  """
  return crypto.sha1_list(content_id, *config.preprocessor)


def ResolveEncodedId(content_id: str, config: corpus_pb2.Corpus) -> str:
  """Compute the hash of a corpus of preprocessed and encoded contentfiles.

  The hash is computed from the ID of the input files and the serialized
  representation of the config proto.
  """
  config_without_contentfiles = corpus_pb2.Corpus()
  config_without_contentfiles.CopyFrom(config)
  # Clear the contentfiles field, since we use the content_id to uniquely
  # identify the input files. This means that corpuses with the same content
  # files delivered through different means (e.g. two separate but identical
  # directories) have the same hash.
  config_without_contentfiles.ClearField('contentfiles')
  return crypto.sha1_list(
      content_id, config_without_contentfiles.SerializeToString())


def GetHashOfArchiveContents(archive: pathlib.Path) -> str:
  """Compute the checksum of the contents of a directory.

  Args:
    archive: Path of the archive.

  Returns:
    Checksum of the archive.

  Raises:
    UserError: If the requested archive does not exist, or cannot be unpacked.
  """
  if not archive.is_file():
    raise errors.UserError(f"Archive not found: '{archive}'")

  with tempfile.TemporaryDirectory(prefix='clgen_corpus_') as d:
    cmd = ['tar', '-xf', str(archive), '-C', d]
    try:
      subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
      raise errors.UserError(f"Archive unpack failed: '{archive}'")
    return checksumdir.dirhash(d, 'sha1')
