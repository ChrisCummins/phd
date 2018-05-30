"""This file defines the logic and training corpuses.

A training corpus is a set of one or more "contentfiles", where each contentfile
is a file containing text to train over.
"""
import os
import pathlib
import pickle
import re
import subprocess
import tempfile
import time
import typing
from tempfile import NamedTemporaryFile

import checksumdir
import humanize
import numpy as np
from absl import logging

from deeplearning.clgen import cache
from deeplearning.clgen import dbutil
from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import features
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import corpus_pb2
from lib.labm8 import crypto
from lib.labm8 import hashcache
from lib.labm8 import lockfile
from lib.labm8 import pbutil
from lib.labm8 import tar


def AssertConfigIsValid(config: corpus_pb2.Corpus) -> corpus_pb2.Corpus:
  try:
    pbutil.AssertFieldIsSet(config, 'contentfiles')
    pbutil.AssertFieldIsSet(config, 'atomizer')
    pbutil.AssertFieldIsSet(config, 'contentfile_separator')
    # Check that the preprocessor pipeline resolves to preprocessor functions.
    [preprocessors.GetPreprocessorFunction(p) for p in config.preprocessor]
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

    cache.cachepath('corpus').mkdir(parents=True, exist_ok=True)
    hc = hashcache.HashCache(
        cache.cachepath('hashcache.db'), 'sha1')
    self.content_id = ResolveContentId(self.config, hc)
    logging.info('Content ID: %s', self.content_id)
    # Database of pre-processed files
    preprocessed_id = ResolvePreprocessedId(self.content_id, self.config)
    cache.cachepath('corpus', 'preprocessed', preprocessed_id).mkdir(
        exist_ok=True, parents=True)
    self.preprocessed = preprocessed.PreprocessedContentFiles(cache.cachepath(
        'corpus', 'preprocessed', preprocessed_id, 'preprocessed.db'))
    logging.info('Preprocessed corpus: %s', preprocessed_id)
    # Data of encoded pre-preprocessed files.
    encoded_id = ResolveEncodedId(self.content_id, self.config)
    cache.cachepath('corpus', 'encoded', encoded_id).mkdir(
        exist_ok=True, parents=True)
    self.encoded = encoded.EncodedContentFiles(cache.cachepath(
        'corpus', 'encoded', encoded_id, 'encoded.db'))
    self.atomizer_path = cache.cachepath(
        'corpus', 'encoded', encoded_id, 'atomizer.pkl')
    logging.info('Encoded corpus: %s', encoded_id)

    self.hash = encoded_id

  def Create(self) -> None:
    """Create the corpus files."""
    preprocessed_lock_path = self.preprocessed.database_path.parent / 'LOCK'
    with lockfile.LockFile(preprocessed_lock_path).acquire(replace_stale=True):
      self.preprocessed.Create(self.config)
    encoded_lock_path = self.encoded.database_path.parent / 'LOCK'
    with lockfile.LockFile(encoded_lock_path).acquire(replace_stale=True):
      self.encoded.Create(self.preprocessed, self.atomizer)

  def ConcatenateTextCorpus(self, shuffle: bool) -> str:
    """Concatenate the entire corpus into a string.

    Args:
      shuffle: If true, randomize order of contentfiles.

    Returns:
      A concatenated corpus string.
    """
    with self.preprocessed.Session() as session:
      return self.config.contentfile_separator.join(
          [x[0] for x in
           session.query(preprocessed.PreprocessedContentFile.text)])

  def GetTrainingData(self, shuffle: bool) -> np.ndarray:
    """Create batches for training.

    Args:
      shuffle: If true, randomize order of contentfiles.

    Returns:
      The encoded corpus.
    """
    # TODO: Join using contentfile_separator.
    # TODO: Can binary numpy strings be concatenated and decoded as one?
    with self.encoded.Session() as session:
      return np.concatenate([
        np.fromstring(x[0]) for x in
        session.query(encoded.EncodedContentFile.data)])

  @property
  def atomizer(self) -> atomizers.AtomizerBase:
    """Must call Create() first."""
    if self._atomizer is None:
      if self.atomizer_path.is_file():
        self._LoadAtomizer()
      else:
        self._CreateAtomizer()
    return self._atomizer

  def _LoadAtomizer(self) -> None:
    with open(self.atomizer_path, 'rb') as infile:
      self._atomizer = pickle.load(infile)

  def _CreateAtomizer(self) -> None:
    """Creates and caches an atomizer."""
    logging.info('Deriving atomizer from preprocessed corpus')
    start_time = time.time()
    corpus_txt = self.ConcatenateTextCorpus(shuffle=False)

    if self.config.HasField('ascii_character_atomizer'):
      self._atomizer = atomizers.AsciiCharacterAtomizer.FromText(corpus_txt)
    elif self.config.HasField('greedy_multichar_atomizer'):
      atoms = set(self.config.greedy_multichar_atomizer.tokens)
      self._atomizer = atomizers.GreedyAtomizer.FromText(corpus_txt, atoms)
    else:
      raise NotImplementedError

    logging.info('%s derived in %s ms',
                 type(self._atomizer).__name__,
                 humanize.intcomma(int((time.time() - start_time) * 1000)))
    with open(self.atomizer_path, 'wb') as f:
      pickle.dump(self.atomizer, f)

  @property
  def vocabulary_size(self) -> int:
    """The number of elements in the corpus vocabulary."""
    return len(self.atomizer.vocab)

  @property
  def size(self) -> int:
    """
    Return the atomized size of the corpus.
    """
    try:
      return self._size
    except AttributeError:
      self.GetTrainingData(False)
      return self._size

  def GetPreprocessedKernels(self, status: int = 0) -> typing.Iterable[str]:
    """
    Return an iterator over all preprocessed kernels.

    Parameters
    ----------
    status : int, optional
        Pre-processed status, {0, 1, 2} for {good, bad, ugly}.

    Returns
    -------
    typing.Iterable[str]
        Sources.
    """
    db = dbutil.connect(self.contentfiles_cache["kernels.db"])
    c = db.cursor()
    query = c.execute(
        f"SELECT Contents FROM PreprocessedFiles WHERE status={status}")
    for row in query.fetchall():
      yield row[0]

  def GetContentFiles(self) -> typing.Iterable[str]:
    """
    Return an iterator over all un-processed samples.

    Returns
    -------
    typing.Iterable[str]
        Samples.
    """
    db = dbutil.connect(self.contentfiles_cache["kernels.db"])
    c = db.cursor()
    query = c.execute("SELECT Contents FROM ContentFiles")
    for row in query.fetchall():
      yield row[0]

  def __repr__(self) -> str:
    nf = dbutil.num_good_kernels(self.contentfiles_cache['kernels.db'])
    return (f'corpus[{self.shorthash}]: {nf} files, {self.size} tokens '
            f'using {self.atomizer}')

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Corpus):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)


def ExpandConfigPath(path: str) -> pathlib.Path:
  return pathlib.Path(os.path.expandvars(path)).expanduser().absolute()


def ResolveContentId(config: corpus_pb2.Corpus, hc: hashcache.HashCache) -> str:
  """Compute the hash of the input contentfiles."""
  if config.HasField('local_directory'):
    try:
      return hc.GetHash(ExpandConfigPath(config.local_directory))
    except FileNotFoundError as e:
      raise errors.UserError(e)
  elif config.HasField('local_tar_archive'):
    return GetHashOfArchiveContents(ExpandConfigPath(config.local_tar_archive))
  else:
    raise NotImplementedError('Unsupported Corpus.contentfiles field value')


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
  with tempfile.TemporaryDirectory(prefix='clgen_corpus_') as d:
    cmd = ['tar', '-xf', str(archive), '-C', d]
    subprocess.check_call(cmd)
    return checksumdir.dirhash(d, 'sha1')


def UnpackDirectoryIfNeeded(path: pathlib.Path) -> pathlib.Path:
  """Unpack a directory tarball and return its path.

  If path is a tarball, unpack it. If path doesn't exist but there is a
  tarball with the same name, unpack it.

  Args:
    path: Path to directory or tarball.

  Returns:
    Path to directory.

  Raises:
    clgen.InternalError: If unable to extract archive.
  """
  if path.is_dir():
    return path

  if path.is_file() and str(path).endswith(".tar.bz2"):
    logging.info('unpacking %s', path)
    tar.unpack_archive(path)
    return pathlib.Path(re.sub(r'.tar.bz2$', '', str(path)))

  if pathlib.Path(str(path) + ".tar.bz2").is_file():
    logging.info('unpacking %s.tar.bz', path)
    tar.unpack_archive(str(path) + ".tar.bz2")
    return path

  raise errors.InternalError(f"Cannot find path '{path}'")


def GetKernelFeatures(code: str, **kwargs) -> np.array:
  """Get features for code.

  Args:
    code: Source code.
    **kwargs: Arguments to features.features()

  Returns:
    An array of feature values.

  Raises:
    FeaturesError: In case of error.
  """
  with NamedTemporaryFile() as outfile:
    outfile.write(code.encode("utf-8"))
    outfile.seek(0)
    f = features.to_np_arrays([outfile.name], **kwargs)
  if len(f) != 1:
    logging.error('features: %s', f)
    raise errors.FeaturesError("code contains more than one kernel")
  return f[0]
