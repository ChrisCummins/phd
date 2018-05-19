#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Manipulating and handling training corpuses.
"""
import codecs
import pickle
import re
from copy import deepcopy
from datetime import datetime
from tempfile import NamedTemporaryFile
from time import time
from typing import Iterable, List, Tuple

import numpy as np

import deeplearning.clgen.clgen.cache
import deeplearning.clgen.clgen.errors
from deeplearning.clgen import clgen
from deeplearning.clgen import dbutil
from deeplearning.clgen import features
from deeplearning.clgen import log
from lib.labm8 import crypto
from lib.labm8 import fs
from lib.labm8 import jsonutil
from lib.labm8 import lockfile
from lib.labm8 import prof
from lib.labm8 import tar
from lib.labm8 import text
from lib.labm8 import types
from lib.labm8.dirhashcache import DirHashCache


# Default options used for corpus. Any values provided by the user will override
# these defaults.
DEFAULT_CORPUS_OPTS = {"created": {"date": str(datetime.now()), }, "eof": False, "batch_size": 50,
                       "seq_length": 50, "vocabulary": "char", "encoding": "default",
                       "preprocess": True, "preserve_order": False, "language": None,
                       # Note no explicit default language.
                       }


class FeaturesError(deeplearning.clgen.clgen.errors.CLgenError):
  """
  Thrown in case of error during features encoding.
  """
  pass


def unpack_directory_if_needed(path: str) -> str:
  """
  If path is a tarball, unpack it. If path doesn't exist but there is a
  tarball with the same name, unpack it.

  Parameters
  ----------
  path : str
      Path to directory or tarball.

  Returns
  -------
  str
      Path to directory.

  Raises
  ------
  clgen.InternalError
      If unable to extract archive.
  """
  if fs.isdir(path):
    return path

  if fs.isfile(path) and path.endswith(".tar.bz2"):
    log.info("unpacking '{}'".format(path))
    tar.unpack_archive(path)
    return re.sub(r'.tar.bz2$', '', path)

  if fs.isfile(path + ".tar.bz2"):
    log.info("unpacking '{}'".format(path + ".tar.bz2"))
    tar.unpack_archive(path + ".tar.bz2")
    return path

  raise deeplearning.clgen.clgen.errors.InternalError(
    "cannot interpret archive '{path}'".format(**vars()))


def get_kernel_features(code: str, **kwargs) -> np.array:
  """
  Get features for code.

  Parameters
  ----------
  code : str
      Source code.
  **kwargs
      Arguments to features.features()

  Returns
  -------
  np.array
      Feature values.
  """
  with NamedTemporaryFile() as outfile:
    outfile.write(code.encode("utf-8"))
    outfile.seek(0)
    f = features.to_np_arrays([outfile.name], **kwargs)
  if len(f) != 1:
    log.error("features:", f)
    raise FeaturesError("code contains more than one kernel")
  return f[0]


def get_cl_kernel_end_idx(src: str, start_idx: int = 0, max_len: int = 5000) -> int:
  """
  Return the index of the character after the end of the OpenCL
  kernel.

  Parameters
  ----------
  src : str
      OpenCL source.
  start_idx : int, optional
      Start index.
  max_len : int, optional
      Maximum kernel length.

  Returns
  -------
  int
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


def get_cl_kernel(src: str, start_idx: int, max_len: int = 5000) -> str:
  """
  Return the OpenCL kernel.

  Parameters
  ----------
  src : str
      OpenCL source.
  start_idx : int, optional
      Start index.
  max_len : int, optional
      Maximum kernel length.

  Returns
  -------
  str
      OpenCL kernel.
  """
  return src[start_idx:get_cl_kernel_end_idx(src, start_idx)]


def get_cl_kernels(src: str) -> List[str]:
  """
  Return OpenCL kernels.

  Parameters
  ----------
  src : str
      OpenCL source.

  Returns
  -------
  List[str]
      OpenCL kernels.
  """
  idxs = text.get_substring_idxs('__kernel', src)
  kernels = [get_cl_kernel(src, i) for i in idxs]
  return kernels


def encode_kernels_db(kernels_db: str, encoding: str) -> None:
  """
  Encode a kernels database.

  Parameters
  ----------
  kernels_db : str
      Path to kernels database.
  encoding : str
      Encoding type.
  """

  def _default(kernels_db: str) -> None:
    pass

  def _static_features(kernels_db: str) -> None:
    log.verbose("Static feature encoding")
    db = dbutil.connect(kernels_db)
    c = db.cursor()
    c.execute("SELECT id,contents FROM PreprocessedFiles WHERE status=0")
    for row in list(c.fetchall()):
      id, contents = row
      c.execute("DELETE FROM PreprocessedFiles WHERE id=?", (id,))
      for i, kernel in enumerate(get_cl_kernels(contents)):
        features = get_kernel_features(kernel)
        kid = "{}-{}".format(id, i)
        if len(features) == 8:
          log.verbose("features", kid)
          feature_str = ("/* {:10} {:10} {:10} {:10} {:10} {:10}"
                         "{:10.3f} {:10.3f} */".format(int(features[0]), int(features[1]),
                                                       int(features[2]), int(features[3]),
                                                       int(features[4]), int(features[5]),
                                                       features[6], features[7]))
          newsource = feature_str + '\n' + kernel
          c.execute("""
                        INSERT INTO PreprocessedFiles (id,contents,status)
                        VALUES (?,?,?)
                    """, (kid, newsource, 0))
        else:
          log.verbose("ignored", kid)
    c.close()
    db.commit()

  # dispatch encoder based on encoding
  encoders = {"default": _default, "static_features": _static_features, }
  encoder = encoders.get(encoding, None)
  if encoder is None:
    raise deeplearning.clgen.clgen.errors.UserError(
      "Unknown encoding type '{bad}'. Supported values: {good}".format(bad=encoding, good=", ".join(
        sorted(encoders.keys()))))
  else:
    encoder(kernels_db)


class Corpus(object):
  """
  Representation of a training corpus.

  Please note corpus instances should be treated as immutable. Upon
  instantiation, a corpus's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, contentid: str, path: str = None, **opts):
    """
    Instantiate a corpus.

    If this is a new corpus, a number of files will be created, which may
    take some time.

    Parameters
    ----------
    contentid : str
        ID of corpus content.
    path : str, optional
        Path to corpus.
    **opts
        Keyword options.
    """
    # Validate options
    for key in opts.keys():
      if key not in DEFAULT_CORPUS_OPTS:
        raise deeplearning.clgen.clgen.errors.UserError(
          "Unsupported corpus option '{}'. Valid keys: {}".format(key, ','.join(
            sorted(DEFAULT_CORPUS_OPTS.keys()))))

    self.opts = deepcopy(DEFAULT_CORPUS_OPTS)
    types.update(self.opts, opts)
    self.opts["id"] = contentid

    # check that contentid exists
    self.language = clgen.Language.from_str(opts.get("language"))
    if (path is None and not fs.isdir(
        deeplearning.clgen.clgen.cache.cachepath("contentfiles", f"{self.language}-{contentid}"))):
      raise deeplearning.clgen.clgen.errors.UserError(
        "corpus {self.language}-{contentid} not found".format(**vars()))

    self.contentid = contentid
    self.contentcache = deeplearning.clgen.clgen.cache.mkcache("contentfiles",
                                                               f"{self.language}-{contentid}")
    self.kernels_db = self.contentcache.keypath('kernels.db')

    self.hash = self._hash(contentid, self.opts)
    self.cache = deeplearning.clgen.clgen.cache.mkcache("corpus", f"{self.language}-{self.hash}")

    log.debug("contentfiles {self.contentid}".format(**vars()))
    log.debug("corpus {hash}".format(hash=self.hash))

    # validate metadata against cache
    self.stats = {"preprocess_time": 0}
    meta = deepcopy(self.to_json())
    if self.cache.get("META"):
      cached_meta = jsonutil.read_file(self.cache["META"])
      self.stats = cached_meta["stats"]  # restore stats

      if "created" in cached_meta:
        del cached_meta["created"]
      del meta["created"]

      if "stats" in cached_meta:
        del cached_meta["stats"]
      del meta["stats"]

      if meta != cached_meta:
        raise deeplearning.clgen.clgen.errors.InternalError("corpus metadata mismatch")
    else:
      self._flush_meta()

    with self.lock.acquire(replace_stale=True):
      self._create_files(path)

  def _flush_meta(self):
    jsonutil.write_file(self.cache.keypath("META"), self.to_json())

  def _create_files(self, path):
    def _init_error(err: Exception, files_to_rm: List[str] = []) -> None:
      """ tidy up in case of error """
      log.error("corpus creation failed. Deleting corpus files")
      for path in files_to_rm:
        if fs.exists(path):
          log.info("removing", path)
          fs.rm(path)
      raise err

    # create kernels database if necessary
    try:
      if path is not None:
        if not fs.isdir(path):
          raise deeplearning.clgen.clgen.errors.UserError(
            "Corpus path '{}' is not a directory".format(path))
        try:
          self.contentcache["kernels.db"]
        except KeyError:
          self._create_kernels_db(path)
    except Exception as e:
      _init_error(e, [self.contentcache.keypath("kernels.db")])

    # preprocess and encode kernel db
    try:
      modified = False
      preprocess_time = time()
      encoding = self.opts["encoding"]
      if clgen.preprocess_db(self.contentcache["kernels.db"], lang=self.language):
        modified = True
        encode_kernels_db(self.contentcache["kernels.db"], encoding)
    except Exception as e:
      _init_error(e)

    if modified:
      preprocess_time = time() - preprocess_time
      self.stats["preprocess_time"] += preprocess_time
      self._flush_meta()

    # create corpus text if not exists
    try:
      try:
        self.cache["corpus.txt"]
      except KeyError:
        self._create_txt()
        assert (self.cache["corpus.txt"])
    except Exception as e:
      _init_error(e, [self.cache.keypath("corpus.txt")])

    # create atomizer if needed
    try:
      try:
        self.cache["atomizer.pkl"]
        self._load_atomizer()
      except KeyError:
        self._create_atomizer(self.opts["vocabulary"])
        assert (self.cache["atomizer.pkl"])
    except Exception as e:
      _init_error(e, [self.cache.keypath("atomizer.pkl")])

  def _hash(self, contentid: str, opts: dict) -> str:
    """ compute corpus hash """
    opts = deepcopy(opts)
    del opts["created"]
    return crypto.sha1_list(contentid, *types.dict_values(opts))

  def _create_kernels_db(self, path: str) -> None:
    """creates and caches kernels.db"""
    log.debug("creating database")

    # create a database and put it in the cache
    tmppath = self.contentcache.keypath("kernels.db.tmp")
    dbutil.create_db(tmppath)
    self.contentcache["kernels.db"] = tmppath

    # get a list of files in the corpus
    filelist = [f for f in fs.ls(path, abspaths=True, recursive=True) if fs.isfile(f)]

    # import files into database
    clgen.fetch(self.contentcache["kernels.db"], filelist)

  def _create_txt(self) -> None:
    """creates and caches corpus.txt"""
    log.debug("creating corpus")

    # TODO: additional options in corpus JSON to accomodate for EOF,
    # different encodings etc.
    tmppath = self.cache.keypath("corpus.txt.tmp")
    dbutil.dump_db(self.contentcache["kernels.db"], tmppath)
    self.cache["corpus.txt"] = tmppath

  def _read_txt(self) -> str:
    with codecs.open(self.cache["corpus.txt"], encoding="utf-8") as infile:
      return infile.read()

  def _create_atomizer(self, vocab: str = "char") -> None:
    """creates and caches atomizer.pkl"""

    def _get_atomizer(corpus_txt: str, vocab: str = "char") -> list:
      """
      Get atomizer for a corpus.

      Parameters
      ----------
      corpus : str
          Corpus.
      vocab : str, optional
          Vocabularly type.

      Returns
      -------
      clgen.Atomizer
          Atomizer.
      """
      atomizers = {"char": clgen.CharacterAtomizer, "greedy": clgen.GreedyAtomizer, }
      atomizerclass = atomizers.get(vocab, None)
      if atomizerclass is None:
        raise deeplearning.clgen.clgen.errors.UserError("Unknown vocabulary type '{bad}'. "
                                                        "Supported values: {good}".format(bad=vocab,
                                                                                          good=", "
                                                                                               "".join(
                                                                                            sorted(
                                                                                              atomizers.keys()))))
      else:
        return atomizerclass.from_text(self.language, corpus_txt)

    log.debug("creating vocab file")
    data = self._read_txt()

    self.atomizer = _get_atomizer(data, vocab)

    self.atoms = self.atomizer.atoms
    self.vocab_size = self.atomizer.vocab_size
    self.vocab = self.atomizer.vocab

    tmp_vocab_file = self.cache.keypath("atomizer.tmp.pkl")
    with open(tmp_vocab_file, 'wb') as f:
      pickle.dump(self.atomizer, f)

    self.cache["atomizer.pkl"] = tmp_vocab_file

  def _load_atomizer(self) -> None:
    with open(self.cache["atomizer.pkl"], 'rb') as infile:
      self.atomizer = pickle.load(infile)

    self.atoms = self.atomizer.atoms
    self.vocab_size = self.atomizer.vocab_size
    self.vocab = self.atomizer.vocab

  def _generate_kernel_corpus(self) -> str:
    """ dump all kernels into a string in a random order """
    db = dbutil.connect(self.contentcache["kernels.db"])
    c = db.cursor()

    # if preservering order, order by line count. Else, order randomly
    orderby = "LC(contents)" if self.opts["preserve_order"] else "RANDOM()"

    c.execute("SELECT PreprocessedFiles.Contents FROM PreprocessedFiles "
              "WHERE status=0 ORDER BY {orderby}".format(orderby=orderby))

    # If file separators are requested, insert EOF markers between files
    sep = '\n\n// EOF\n\n' if self.opts["eof"] else '\n\n'

    return sep.join(row[0] for row in c.fetchall())

  def create_batches(self) -> None:
    """
    Create batches for training.
    """
    self.reset_batch_pointer()

    # generate a kernel corpus
    data = self._generate_kernel_corpus()

    # encode corpus into vocab indices
    self._tensor = self.atomizer.atomize(data)

    batch_size = self.batch_size
    seq_length = self.seq_length

    # set corpus size and number of batches
    self._size = len(self._tensor)
    self._num_batches = int(self.size / (batch_size * seq_length))
    if self.num_batches == 0:
      raise deeplearning.clgen.clgen.errors.UserError(
        "Not enough data. Use a smaller seq_length and batch_size")

    # split into batches
    self._tensor = self._tensor[:self.num_batches * batch_size * seq_length]
    xdata = self._tensor
    ydata = np.copy(self._tensor)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    self._x_batches = np.split(xdata.reshape(batch_size, -1), self.num_batches, 1)
    self._y_batches = np.split(ydata.reshape(batch_size, -1), self.num_batches, 1)

  @property
  def shorthash(self):
    return clgen._shorthash(self.hash, deeplearning.clgen.clgen.cache.cachepath("corpus"))

  @property
  def lock(self):
    lockpath = self.cache.keypath("LOCK")
    return lockfile.LockFile(lockpath)

  @property
  def batch_size(self) -> int:
    return self.opts["batch_size"]

  @property
  def seq_length(self) -> int:
    return self.opts["seq_length"]

  @property
  def size(self) -> int:
    """
    Return the atomized size of the corpus.
    """
    try:
      return self._size
    except AttributeError:
      self.create_batches()
      return self._size

  @property
  def num_batches(self) -> int:
    try:
      return self._num_batches
    except AttributeError:
      self.create_batches()
      return self._num_batches

  def reset_batch_pointer(self) -> None:
    """
    Resets batch pointer to first batch.
    """
    self._pointer = 0

  def next_batch(self) -> Tuple[np.array, np.array]:
    """
    Fetch next batch indices.

    Returns
    -------
    Tuple[np.array, np.array]
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

  def preprocessed(self, status: int = 0) -> Iterable[str]:
    """
    Return an iterator over all preprocessed kernels.

    Parameters
    ----------
    status : int, optional
        Pre-processed status, {0, 1, 2} for {good, bad, ugly}.

    Returns
    -------
    Iterable[str]
        Sources.
    """
    db = dbutil.connect(self.contentcache["kernels.db"])
    c = db.cursor()
    query = c.execute(
      "SELECT Contents FROM PreprocessedFiles WHERE status={status}".format(**vars()))
    for row in query.fetchall():
      yield row[0]

  def contentfiles(self) -> Iterable[str]:
    """
    Return an iterator over all un-processed samples.

    Returns
    -------
    Iterable[str]
        Samples.
    """
    db = dbutil.connect(self.contentcache["kernels.db"])
    c = db.cursor()
    query = c.execute("SELECT Contents FROM ContentFiles")
    for row in query.fetchall():
      yield row[0]

  def __repr__(self) -> str:
    nf = dbutil.num_good_kernels(self.contentcache['kernels.db'])
    return (f"corpus[{self.shorthash}]: {nf} files, {self.size} tokens " + f"using {self.opts["
                                                                           f"'vocabulary']} "
                                                                           f"vocabulary of size "
            + f"{self.atomizer.vocab_size}")

  def to_json(self) -> dict:
    d = deepcopy(self.opts)
    d["contentfiles"] = self.contentid
    d["stats"] = self.stats
    return d

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Corpus):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)

  @staticmethod
  def from_json(corpus_json: dict) -> 'Corpus':
    """
    Instantiate Corpus from JSON.

    Parameters
    ----------
    corpus_json : dict
        Specification.

    Returns
    -------
    Corpus
        Insantiated corpus.
    """
    path = corpus_json.pop("path", None)
    uid = corpus_json.pop("id", None)
    language = clgen.Language.from_str(corpus_json.get("language"))

    if path:
      path = unpack_directory_if_needed(fs.abspath(path))
      if not fs.isdir(path):
        raise deeplearning.clgen.clgen.errors.UserError(
          "Corpus path '{}' is not a directory".format(path))

      dirhashcache = DirHashCache(deeplearning.clgen.clgen.cache.cachepath("dirhash.db"), 'sha1')
      uid = prof.profile(dirhashcache.dirhash, path)
    elif uid:
      cache_path = deeplearning.clgen.clgen.cache.mkcache("contentfiles", f"{language}-{uid}").path
      if not fs.isdir(cache_path):
        raise deeplearning.clgen.clgen.errors.UserError("Corpus content {} not found".format(uid))
    else:
      raise deeplearning.clgen.clgen.errors.UserError("No corpus path or ID provided")

    if "stats" in corpus_json:  # ignore stats
      del corpus_json["stats"]

    if "contentfiles" in corpus_json:
      del corpus_json["contentfiles"]

    return prof.profile(Corpus, uid, path=path, **corpus_json)
