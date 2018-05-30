"""This file defines the logic and training corpuses.

A training corpus is a set of one or more "contentfiles", where each contentfile
is a file containing text to train over.
"""
import codecs
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
from deeplearning.clgen.corpuses import fetch
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import internal_pb2
from lib.labm8 import crypto
from lib.labm8 import fs
from lib.labm8 import hashcache
from lib.labm8 import lockfile
from lib.labm8 import pbutil
from lib.labm8 import prof
from lib.labm8 import tar
from lib.labm8 import text


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
    self.config.CopyFrom(config)

    cache.cachepath('corpus').mkdir(parents=True, exist_ok=True)
    hc = hashcache.HashCache(
        cache.cachepath('corpus', 'hashcache.db'), 'sha1')
    self.content_id = ResolveContentId(self.config, hc)
    logging.info('Content ID: %s', self.content_id)
    # Database of pre-processed files
    self.preprocessed_id = ResolvePreprocessedId(self.content_id, self.config)
    cache.cachepath('corpus', 'preprocessed', self.preprocessed_id).mkdir(
        exist_ok=True, parents=True)
    self.preprocessed = preprocessed.PreprocessedContentFiles(cache.cachepath(
        'corpus', 'preprocessed', self.preprocessed_id, 'preprocessed.db'))
    logging.info('Preprocessed corpus: %s', self.preprocessed_id)
    # Data of encoded pre-preprocessed files.
    self.encoded_id = ResolvePreprocessedId(self.content_id, self.config)
    cache.cachepath('corpus', 'encoded', self.encoded_id).mkdir(
        exist_ok=True, parents=True)
    self.encoded = encoded.EncodedContentFiles(cache.cachepath(
        'corpus', 'encoded', self.encoded_id, 'encoded.db'))
    logging.info('Encoded corpus: %s', self.encoded_id)

    # Determine the corpus cache path. This will depend on whether a path or
    # an id was specified.
    path = None
    if config.HasField('local_directory'):
      path = pathlib.Path(os.path.expandvars(config.local_directory)).absolute()
      path = UnpackDirectoryIfNeeded(path)
      if not fs.isdir(path):
        raise errors.UserError(
            "Corpus path '{}' is not a directory".format(path))
      if fs.directory_is_empty(path):
        raise errors.EmptyCorpusException(
            f"Contentfiles directory is empty: '{path}'")
      hasher = hashcache.HashCache(cache.cachepath("dirhash.db"), 'sha1')
      self.content_id = prof.profile(hasher.GetHash, path)
    elif config.HasField('id'):
      self.content_id = config.id
      if not fs.isdir(
          cache.cachepath("contentfiles", config.id)):
        raise errors.UserError(f"Corpus not found: '{config.id}'")
    else:
      raise errors.UserError('Must specify corpus id or path.')

    self.contentfiles_cache = cache.mkcache("contentfiles", self.content_id)
    self.kernels_db = self.contentfiles_cache.keypath('kernels.db')
    self.hash = ResolveEncodedId(self.content_id, config)
    self.cache = cache.mkcache("corpus", self.hash)

    logging.debug('contentfiles %s', self.content_id)
    logging.debug('corpus %s', self.hash)

    # Validate metadata against cache.
    if self.cache.get('META.pbtxt'):
      cached_meta = pbutil.FromFile(pathlib.Path(self.cache['META.pbtxt']),
                                    internal_pb2.CorpusMeta())
      if config != cached_meta.config:
        raise errors.InternalError('Metadata mismatch')
      self.meta = cached_meta
    else:
      self.meta = internal_pb2.CorpusMeta()
      self.meta.config.CopyFrom(self.config)
      self._FlushMeta()

    with path and self.lock.acquire(replace_stale=True):
      self._CreateCorpusFiles(path)

  def Create(self) -> None:
    """Create the corpus files."""
    self.preprocessed.Create()
    self.encoded.Create()

  def _FlushMeta(self):
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath('META.pbtxt')))

  def _CreateCorpusFiles(self, contentfiles_dir: pathlib.Path) -> None:
    """Perform the initial creation of derived corpus files.

    Args:
      contentfiles_dir: The path to the corpus contentfiles directory.
    """

    def _Error(err: Exception, files_to_rm: typing.List[pathlib.Path]) -> None:
      """Tidy up in case of error."""
      logging.error('corpus creation failed. Deleting corpus files')
      for path in files_to_rm:
        if path.is_file():
          logging.info('removing %s', path)
          fs.rm(path)
      raise err

    # Create the kernels database if necessary.
    try:
      self.contentfiles_cache["kernels.db"]
    except KeyError:
      self._CreateKernelsDatabase(contentfiles_dir)

    # Preprocess the kernels database, if necessary.
    modified = False
    if self.config.preprocessor:
      try:
        preprocess_time = time.time()
        if preprocessors.PreprocessDatabase(
            pathlib.Path(self.contentfiles_cache["kernels.db"]),
            self.config.preprocessor):
          modified = True
      except Exception as e:
        _Error(e, [])
    else:
      # If we have no preprocessors to run, simply copy the ContentFiles over to
      # PreprocessedFiles, rather than spooling up all of the preprocessors.
      db = dbutil.connect(self.contentfiles_cache["kernels.db"])
      c = db.cursor()
      c.execute("""
INSERT INTO PreprocessedFiles (id, contents, status)
SELECT id, contents, 0
FROM ContentFiles
WHERE ContentFiles.id NOT IN (
  SELECT id
  FROM PreprocessedFiles
)
""")
      db.commit()
      c.close()
      db.close()

    # TODO(cec): Raise an error if there are no preprocessed kernels with
    # status 0.

    if modified:
      preprocess_time = time.time() - preprocess_time
      self.meta.preprocess_time_ms += int(preprocess_time * 1000)
      self._FlushMeta()

    # Create the corpus text if it does not exist.
    try:
      try:
        self.cache["corpus.txt"]
      except KeyError:
        self._CreateCorpusText()
        assert self.cache["corpus.txt"]
    except Exception as e:
      _Error(e, [pathlib.Path(self.cache.keypath("corpus.txt"))])

    # Create the atomizer if needed.
    try:
      try:
        self._LoadAtomizer(pathlib.Path(self.cache["atomizer.pkl"]))
      except KeyError:
        self._CreateAtomizer()
        assert self.cache["atomizer.pkl"]
    except Exception as e:
      _Error(e, [pathlib.Path(self.cache.keypath("atomizer.pkl"))])

  def _CreateKernelsDatabase(self, path: pathlib.Path) -> None:
    """creates and caches kernels.db"""
    logging.debug('creating database')

    # create a database and put it in the cache
    tmppath = self.contentfiles_cache.keypath("kernels.db.tmp")
    dbutil.create_db(tmppath)
    self.contentfiles_cache["kernels.db"] = tmppath

    # get a list of files in the corpus
    filelist = [f for f in fs.ls(path, abspaths=True, recursive=True) if
                fs.isfile(f)]

    # import files into database
    fetch.fetch(self.contentfiles_cache["kernels.db"], filelist)

  def _CreateCorpusText(self) -> None:
    """creates and caches corpus.txt"""
    logging.debug('creating corpus')
    with open(self.cache.keypath('corpus.txt'), 'w') as f:
      f.write(self.ConcatenateTextCorpus(shuffle=True))

  def _ReadCorpusTxt(self) -> str:
    with codecs.open(self.cache["corpus.txt"], encoding="utf-8") as infile:
      return infile.read()

  def _CreateAtomizer(self) -> None:
    """creates and caches atomizer.pkl"""

    logging.debug('creating vocab file')
    corpus_txt = self._ReadCorpusTxt()

    if self.config.HasField('ascii_character_atomizer'):
      self.atomizer = atomizers.AsciiCharacterAtomizer.FromText(corpus_txt)
    elif self.config.HasField('greedy_multichar_atomizer'):
      atoms = set(self.config.greedy_multichar_atomizer.tokens)
      self.atomizer = atomizers.GreedyAtomizer.FromText(corpus_txt, atoms)
    else:
      raise errors.UserError('No atomizer specified')

    self.atoms = self.atomizer.atoms
    self.vocab_size = self.atomizer.vocab_size
    self.vocab = self.atomizer.vocab

    tmp_vocab_file = self.cache.keypath("atomizer.tmp.pkl")
    with open(tmp_vocab_file, 'wb') as f:
      pickle.dump(self.atomizer, f)

    self.cache["atomizer.pkl"] = tmp_vocab_file

  def _LoadAtomizer(self, atomizer_path: pathlib.Path) -> None:
    with open(atomizer_path, 'rb') as infile:
      self.atomizer = pickle.load(infile)

    self.atoms = self.atomizer.atoms
    self.vocab_size = self.atomizer.vocab_size
    self.vocab = self.atomizer.vocab

  def ConcatenateTextCorpus(self, shuffle: bool) -> str:
    """Concatenate the entire corpus into a string.

    Args:
      shuffle: If true, randomize order of contentfiles.

    Returns:
      A concatenated corpus string.
    """
    db = dbutil.connect(self.contentfiles_cache["kernels.db"])
    c = db.cursor()
    order_by = 'RANDOM()' if shuffle else 'LENGTH(contents) DESC'
    c.execute('SELECT contents FROM PreprocessedFiles WHERE status=0 '
              f'ORDER BY {order_by}')
    sep = self.config.contentfile_separator or '\n\n'
    return sep.join(row[0] for row in c.fetchall())

  def GetTrainingData(self, shuffle: bool) -> np.ndarray:
    """Create batches for training.

    Args:
      shuffle: If true, randomize order of contentfiles.

    Returns:
      The encoded corpus.
    """
    start_time = time.time()
    # Generate a corpus by randomly shuffling the contentfiles.
    corpus_text = self.ConcatenateTextCorpus(shuffle)
    # Encode the corpus into an array of encoded tokens.
    tokenized_corpus = self.atomizer.AtomizeString(corpus_text)
    # Set the corpus size as the number of tokens.
    num_tokens = len(tokenized_corpus)
    self._size = num_tokens
    logging.info('%s derived %s token corpus of length %s in %s ms',
                 type(self.atomizer).__name__,
                 humanize.intcomma(len(self.atomizer.vocab)),
                 humanize.intcomma(len(tokenized_corpus)),
                 humanize.intcomma(
                     int(round((time.time() - start_time) * 1000))))
    return tokenized_corpus

  @property
  def shorthash(self):
    return cache.ShortHash(self.hash, cache.cachepath("corpus"))

  @property
  def lock(self):
    lockpath = self.cache.keypath("LOCK")
    return lockfile.LockFile(lockpath)

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
  with tempfile.TemporaryDirectory() as d:
    cmd = ['tar', '-xf', str(archive), '-C', d]
    subprocess.check_call(cmd)
    return checksumdir(d, 'sha1')


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


def GetClKernelEndIndex(src: str, start_idx: int = 0,
                        max_len: int = 5000) -> int:
  """Return the index of the character after the end of the OpenCL kernel.

  Args:
    src: OpenCL source.
    start_idx: Start index.
    max_len: Maximum kernel length.

  Returns:
    Index of end of OpenCL kernel.
  """
  i = src.find('{', start_idx) + 1
  d = 1  # depth
  while i < min(len(src), start_idx + max_len) and d > 0:
    if src[i] == '{':
      d += 1
    elif src[i] == '}':
      d -= 1
    i += 1
  return i


def GetClKernel(src: str, start_idx: int, max_len: int = 5000) -> str:
  """Return the OpenCL kernel.

  Args:
    src: OpenCL source.
    start_idx: Start index.
    max_len: Maximum kernel length.

  Returns:
    OpenCL kernel.
  """
  return src[start_idx:GetClKernelEndIndex(src, start_idx, max_len)]


def GetClKernels(src: str) -> typing.List[str]:
  """
  Return OpenCL kernels.

  Args:
    src: OpenCL source.

  Returns:
    OpenCL kernels.
  """
  idxs = text.get_substring_idxs('__kernel', src)
  kernels = [GetClKernel(src, i) for i in idxs]
  return kernels
