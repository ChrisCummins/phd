#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
import re
import codecs
import numpy as np

from checksumdir import dirhash
from collections import Counter
from copy import deepcopy
from labm8 import fs
from six.moves import cPickle
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

import clgen
from clgen import atomizer
from clgen import cache
from clgen import clutil
from clgen import dbutil
from clgen import explore
from clgen import fetch
from clgen import log
from clgen import preprocess
from clgen.cache import Cache
from clgen.train import train


# Default options used for corpus. Any values provided by the user will override
# these defaults.
DEFAULT_CORPUS_OPTS = {
    "eof": False,
    "batch_size": 50,
    "seq_length": 50,
    "vocabulary": "char",
    "encoding": "default",
    "preserve_order": False,
}


def unpack_directory_if_needed(path):
    """
    If path is a tarball, unpack it. If path doesn't exist but there is a
    tarball with the same name, unpack it.

    Arguments:
        path (str): Path to directory or tarball.

    Returns:
        str: Path to directory.
    """
    if fs.isdir(path):
        return path

    if fs.isfile(path) and path.endswith(".tar.bz2"):
        log.info("unpacking '{}'".format(path))
        clgen.unpack_archive(path)
        return re.sub(r'.tar.bz2$', '', path)

    if fs.isfile(path + ".tar.bz2"):
        log.info("unpacking '{}'".format(path + ".tar.bz2"))
        clgen.unpack_archive(path + ".tar.bz2")
        return path

    return path


def get_atomizer(corpus: str, vocab: str="char") -> list:
    """
    Get atomizer for a corpus.

    Arguments:
        corpus (str): Corpus.
        vocab (str, optional): Vocabularly type.

    Returns:
        atomizer.Atomizer: Atomizer.
    """
    atomizers = {
        "char": atomizer.CharacterAtomizer,
        "greedy": atomizer.GreedyAtomizer,
    }
    atomizerclass = atomizers.get(vocab, None)
    if atomizerclass is None:
        raise clgen.UserError(
            "Unknown vocabulary type '{bad}'. Supported values: {good}".format(
                bad=vocab, good=", ".join(sorted(atomizers.keys()))))
    else:
        return atomizerclass.from_text(corpus)


def features_from_file(path):
    """
    Fetch features from file.

    Arguments:
        path (str): Path to file.

    Returns:
        np.array: Feature values.
    """
    # hacky call to clgen-features and parse output
    cmd = ['clgen-features', path]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    cout, _ = proc.communicate()
    features = [float(x) for x in
                cout.decode('utf-8').split('\n')[1].split(',')[2:]]
    return np.array(features)


def get_features(code):
    """
    Get features for code.

    Arguments:
        code (str): Source code.

    Returns:
        np.array: Feature values.
    """
    with NamedTemporaryFile() as outfile:
        outfile.write(code.encode("utf-8"))
        outfile.seek(0)
        features = features_from_file(outfile.name)
    return features


def encode(kernels_db: str, encoding: str) -> None:
    """
    Encode a kernels database.

    Arguments:
        kernels_db (str): Path to kernels database.
        encoding (str): Encoding type.
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
            for i, kernel in enumerate(clutil.get_cl_kernels(contents)):
                features = get_features(kernel)
                kid = "{}-{}".format(id, i)
                log.verbose("features", kid)
                if len(features) == 8:
                    feature_str = ("/* {:10} {:10} {:10} {:10} {:10} {:10}"
                                   "{:10.3f} {:10.3f} */".format(
                                       int(features[0]),
                                       int(features[1]),
                                       int(features[2]),
                                       int(features[3]),
                                       int(features[4]),
                                       int(features[5]),
                                       features[6],
                                       features[7]))
                    newsource = feature_str + '\n' + kernel
                    c.execute("""
                        INSERT INTO PreprocessedFiles (id,contents,status)
                        VALUES (?,?,?)
                    """, (kid, newsource, 0))
        c.close()
        db.commit()

    # dispatch encoder based on encoding
    encoders = {
        "default": _default,
        "static_features": _static_features,
    }
    encoder = encoders.get(encoding, None)
    if encoder is None:
        raise clgen.UserError(
            "Unknown encoding type '{bad}'. Supported values: {good}".format(
                bad=encoding, good=", ".join(sorted(encoders.keys()))))
    else:
        encoder(kernels_db)


class Corpus(clgen.CLgenObject):
    """
    Representation of a training corpus.
    """
    def __init__(self, contentid: str, path: str=None, **opts):
        """
        Instantiate a corpus.

        If this is a new corpus, a number of files will be created, which may
        take some time.

        Arguments:
            contentid (str): ID of corpus content.
            path (str, optional): Path to corpus.
            **opts: Keyword options.
        """
        def _init_error(err):
            """ tidy up in case of error """
            log.error("corpus creation failed. Deleting corpus files")
            paths = [
                fs.path(self.cache.path, "kernels.db"),
                fs.path(self.cache.path, "corpus.txt"),
                fs.path(self.cache.path, "tensor.npy"),
                fs.path(self.cache.path, "atomizer.pkl")
            ]
            for path in paths:
                if fs.exists(path):
                    log.info("removing", path)
                    fs.rm(path)
            raise err

        # Validate options
        for key in opts.keys():
            if key not in DEFAULT_CORPUS_OPTS:
                raise clgen.UserError(
                    "Unsupported corpus option '{}'. Valid keys: {}".format(
                        key, ','.join(sorted(DEFAULT_CORPUS_OPTS.keys()))))

        self.opts = deepcopy(DEFAULT_CORPUS_OPTS)
        clgen.update(self.opts, opts)
        self.hash = self._hash(contentid, self.opts)
        self.cache = Cache(fs.path("corpus", self.hash))

        log.debug("corpus {hash}".format(hash=self.hash))

        try:
            if path is not None:
                if not fs.isdir(path):
                    raise clgen.UserError(
                        "Corpus path '{}' is not a directory".format(path))
                # create kernels database if necessary
                if not self.cache["kernels.db"]:
                    self._create_kernels_db(path, self.opts["encoding"])
                    assert(self.cache["kernels.db"])

            # create corpus text if not exists
            if not self.cache["corpus.txt"]:
                self._create_txt()
                assert(self.cache["corpus.txt"])

            # create atomizer if needed
            if self.cache["atomizer.pkl"]:
                self._load_atomizer()
                assert(self.cache["atomizer.pkl"])
            else:
                self._create_atomizer(self.opts["vocabulary"])
        except Exception as e:
            _init_error(e)

    def _hash(self, contentid: str, opts: dict) -> str:
        """ compute corpus hash """
        return clgen.checksum_list(contentid, *clgen.dict_values(opts))

    def _create_kernels_db(self, path, encoding="default"):
        """creates and caches kernels.db"""
        log.debug("creating database")

        # create a database and put it in the cache
        tmppath = fs.path(self.cache.path, "kernels.db.tmp")
        dbutil.create_db(tmppath)
        self.cache["kernels.db"] = tmppath

        # get a list of files in the corpus
        filelist = [f for f in fs.ls(path, abspaths=True, recursive=True)
                    if fs.isfile(f)]

        # import files into database
        fetch.fetch_fs(self.cache["kernels.db"], filelist)

        # preprocess files
        preprocess.preprocess_db(self.cache["kernels.db"])

        # encode kernel db
        encode(self.cache["kernels.db"], encoding)

        # print database stats
        explore.explore(self.cache["kernels.db"])

    def _create_txt(self):
        """creates and caches corpus.txt"""
        log.debug("creating corpus")

        # TODO: additional options in corpus JSON to accomodate for EOF,
        # different encodings etc.
        tmppath = fs.path(self.cache.path, "corpus.txt.tmp")
        train(self.cache["kernels.db"], tmppath)
        self.cache["corpus.txt"] = tmppath

    def _read_txt(self):
        with codecs.open(self.cache["corpus.txt"], encoding="utf-8") as infile:
            return infile.read()

    def _create_atomizer(self, vocab="char"):
        """creates and caches atomizer.pkl"""
        log.debug("creating vocab file")

        data = self._read_txt()

        self.atomizer = get_atomizer(data, vocab)

        self.atoms = self.atomizer.atoms
        self.vocab_size = self.atomizer.vocab_size
        self.vocab = self.atomizer.vocab

        tmp_vocab_file = fs.path(self.cache.path, "atomizer.tmp.pkl")
        with open(tmp_vocab_file, 'wb') as f:
            cPickle.dump(self.atomizer, f)

        self.cache["atomizer.pkl"] = tmp_vocab_file

    def _load_atomizer(self):
        with open(self.cache["atomizer.pkl"], 'rb') as infile:
            self.atomizer = cPickle.load(infile)

        self.atoms = self.atomizer.atoms
        self.vocab_size = self.atomizer.vocab_size
        self.vocab = self.atomizer.vocab

    def _generate_kernel_corpus(self) -> str:
        """ dump all kernels into a string in a random order """
        db = dbutil.connect(self.cache["kernels.db"])
        c = db.cursor()

        # if preservering order, order by line count. Else, order randomly
        orderby = "LC(contents)" if self.opts["preserve_order"] else "RANDOM()"

        c.execute("SELECT PreprocessedFiles.Contents FROM PreprocessedFiles "
                  "WHERE status=0 ORDER BY {orderby}".format(orderby=orderby))

        # If file separators are requested, insert EOF markers between files
        sep = '\n\n// EOF\n\n' if self.opts["eof"] else '\n\n'

        return sep.join(row[0] for row in c.fetchall())

    def create_batches(self):
        """
        Create batches for training.
        """
        log.debug("creating batches")
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
            raise clgen.UserError(
                "Not enough data. Use a smaller seq_length and batch_size")

        # split into batches
        self._tensor = self._tensor[:self.num_batches * batch_size * seq_length]
        xdata = self._tensor
        ydata = np.copy(self._tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self._x_batches = np.split(xdata.reshape(batch_size, -1),
                                   self.num_batches, 1)
        self._y_batches = np.split(ydata.reshape(batch_size, -1),
                                   self.num_batches, 1)

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

    @property
    def meta(self):
        """
        Get corpus metadata.

        Returns:
            dict: Metadata.
        """
        _meta = deepcopy(self.opts)
        _meta["id"] = self.hash
        return _meta

    def reset_batch_pointer(self):
        """
        Resets batch pointer to first batch.
        """
        self._pointer = 0

    def next_batch(self):
        """
        Fetch next batch indices.

        Returns:
            (np.array, np.array): X, Y batch tuple.
        """
        x = self._x_batches[self._pointer]
        y = self._y_batches[self._pointer]
        self._pointer += 1
        return x, y

    def set_batch_pointer(self, pointer):
        """
        Set batch pointer.

        Arguments:
            pointer (int): New batch pointer.
        """
        self._pointer = pointer

    def __repr__(self):
        n = dbutil.num_good_kernels(self.cache['kernels.db'])
        return "corpus of {n} files".format(n=n)

    @staticmethod
    def from_json(corpus_json):
        """
        Instantiate Corpus from JSON.

        Arguments:
            corpus_json (dict): Specification.

        Returns:
            Corpus: Insantiated corpus.
        """
        path = corpus_json.pop("path", None)
        uid = corpus_json.pop("id", None)

        if path:
            path = unpack_directory_if_needed(fs.abspath(path))
            if not fs.isdir(path):
                raise clgen.UserError(
                    "Corpus path '{}' is not a directory".format(path))
            uid = dirhash(path, 'sha1')
        elif uid:
            cache_path = fs.path(cache.ROOT, "corpus", uid)
            if not fs.isdir(cache_path):
                raise clgen.UserError("Corpus {} not found".format(uid))
        else:
            raise clgen.UserError("No corpus path or ID provided")

        return Corpus(uid, path=path, **corpus_json)


def preprocessed_kernels(corpus):
    """
    Return an iterator over all preprocessed kernels.

    Arguments:
        corpus (Corpus): Corpus.

    Returns:
        sequence of str: Kernel sources.
    """
    assert(isinstance(corpus, Corpus))
    db = dbutil.connect(corpus.cache["kernels.db"])
    c = db.cursor()
    query = c.execute("SELECT Contents FROM PreprocessedFiles WHERE status=0")
    for row in query.fetchall():
        yield row[0]


def most_common_prototypes(c, n):
    """
    Return the n most frequently occuring prototypes.

    Arguments:
        c (Corpus): Corpus.
        n (int): Number of prototypes to return:

    Returns:
        tuple of list of tuples, int:
    """
    from clgen import clutil

    prototypes = []
    for kernel in preprocessed_kernels(c):
        try:
            prototype = clutil.KernelPrototype.from_source(kernel)
            if prototype.is_synthesizable:
                prototypes.append(", ".join(str(x) for x in prototype.args))
        except clutil.PrototypeException:
            pass

    # Convert frequency into ratios
    counter = Counter(prototypes)
    results = []
    for row in counter.most_common(n):
        prototype, freq = row
        ratio = freq / len(prototypes)
        results.append((ratio, prototype))

    return results, len(prototypes)
