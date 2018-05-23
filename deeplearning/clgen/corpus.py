"""This file defines the logic and training corpuses.

A training corpus is a set of one or more "contentfiles", where each contentfile
is a file containing text to train over.
"""
import codecs
import math
import pathlib
import pickle
import re
import typing
from tempfile import NamedTemporaryFile
from time import time

import numpy as np
from absl import logging

from deeplearning.clgen import atomizers
from deeplearning.clgen import cache
from deeplearning.clgen import dbutil
from deeplearning.clgen import errors
from deeplearning.clgen import features
from deeplearning.clgen import fetch
from deeplearning.clgen import languages
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import internal_pb2
from lib.labm8 import crypto
from lib.labm8 import dirhashcache
from lib.labm8 import fs
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
      UserError: In case the corpus is not found, or config contains invalid
        options.
      EmptyCorpusException: In case the corpus contains no data.
    """
    # Make a local copy of the configuration.
    self.config = corpus_pb2.Corpus()
    self.config.CopyFrom(config)

    self.language = languages.Language.from_str(config.language)

    # Validate config options.
    if config.sequence_length < 1:
      raise errors.UserError('Corpus.sequence_length must be >= 1')

    # Determine the corpus cache path. This will depend on whether a path or
    # an id was specified.
    path = None
    if config.HasField('path'):
      path = UnpackDirectoryIfNeeded(pathlib.Path(config.path).absolute())
      if not fs.isdir(path):
        raise errors.UserError(
          "Corpus path '{}' is not a directory".format(path))
      if fs.directory_is_empty(path):
        raise errors.EmptyCorpusException(f"Corpus path '{path}' is empty")
      hasher = dirhashcache.DirHashCache(cache.cachepath("dirhash.db"), 'sha1')
      self.content_id = prof.profile(hasher.dirhash, path)
    elif config.HasField('id'):
      self.content_id = config.id
      if not fs.isdir(
          cache.cachepath("contentfiles", f"{self.language}-{config.id}")):
        raise errors.UserError(f"corpus {self.language}-{config.id} not found")
    else:
      raise errors.UserError('Must specify corpus id or path.')

    cache_name = f"{self.language}-{self.content_id}"
    self.contentfiles_cache = cache.mkcache("contentfiles", cache_name)
    self.kernels_db = self.contentfiles_cache.keypath('kernels.db')
    self.hash = self._ComputeHash(self.content_id, config)
    self.cache = cache.mkcache("corpus", f"{self.language}-{self.hash}")

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
        preprocess_time = time()
        if preprocessors.PreprocessDatabase(
            pathlib.Path(self.contentfiles_cache["kernels.db"]), self.language,
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
      preprocess_time = time() - preprocess_time
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

  @staticmethod
  def _ComputeHash(contentid: str, config: corpus_pb2.Corpus) -> str:
    """Compute the hash of a corpus.

    The hash is computed from the ID of the contentfiles and the serialized
    representation of the config proto.
    """
    config_without_contentfiles = corpus_pb2.Corpus()
    config_without_contentfiles.CopyFrom(config)
    config_without_contentfiles.ClearField('contentfiles')
    return crypto.sha1_list(contentid,
                            config_without_contentfiles.SerializeToString())

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

    # TODO: additional options in corpus JSON to accomodate for EOF,
    # different encodings etc.
    tmppath = self.cache.keypath("corpus.txt.tmp")
    dbutil.dump_db(self.contentfiles_cache["kernels.db"], tmppath)
    self.cache["corpus.txt"] = tmppath

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

  def CreateBatches(self, batch_size: int, shuffle: bool) -> None:
    """Create batches for training.

    Args:
      shuffle: If true, randomize order of contentfiles.
    """
    self.ResetBatchPointer()

    # generate a kernel corpus
    data = self.ConcatenateTextCorpus(shuffle)

    # encode corpus into vocab indices
    self._tensor = self.atomizer.AtomizeString(data)

    # set corpus size and number of batches
    self._size = len(self._tensor)
    # TODO(cec): Investigate this. Use math.floor() seems one batch too few?
    self._num_batches = math.floor(self.size / (batch_size * self.seq_length))
    if not self.num_batches:
      raise errors.UserError(
        "Not enough data. Use a smaller seq_length and batch_size. "
        f'Current data size = {self.size}, seq_length = {self.seq_length}, and '
        f'batch_size {batch_size}.')

    # split into batches
    self._tensor = self._tensor[
                   :self.num_batches * batch_size * self.seq_length]
    xdata = self._tensor
    ydata = np.copy(self._tensor)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    self._x_batches = np.split(xdata.reshape(batch_size, -1), self.num_batches,
                               1)
    self._y_batches = np.split(ydata.reshape(batch_size, -1), self.num_batches,
                               1)

  @property
  def shorthash(self):
    return cache.ShortHash(self.hash, cache.cachepath("corpus"))

  @property
  def lock(self):
    lockpath = self.cache.keypath("LOCK")
    return lockfile.LockFile(lockpath)

  @property
  def seq_length(self) -> int:
    return self.config.sequence_length

  @property
  def size(self) -> int:
    """
    Return the atomized size of the corpus.
    """
    try:
      return self._size
    except AttributeError:
      self._size = self.ConcatenateTextCorpus(False)
      return self._size

  @property
  def num_batches(self) -> int:
    try:
      return self._num_batches
    except AttributeError:
      self.CreateBatches(1, False)
      return self._num_batches

  def ResetBatchPointer(self) -> None:
    """
    Resets batch pointer to first batch.
    """
    self._pointer = 0

  def next_batch(self) -> typing.Tuple[np.array, np.array]:
    """
    Fetch next batch indices.

    Returns
    -------
    typing.Tuple[np.array, np.array]
        X, Y batch tuple.
    """
    x = self._x_batches[self._pointer]
    y = self._y_batches[self._pointer]
    self._pointer += 1
    return x, y

  def set_batch_pointer(self, pointer: int) -> None:
    """
    Set batch pointer.

    Parameters
    ----------
    pointer : int
        New batch pointer.
    """
    self._pointer = pointer

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
