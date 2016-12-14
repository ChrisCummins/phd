#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
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

from collections import Counter
from checksumdir import dirhash
from labm8 import fs
from six.moves import cPickle

import clgen
from clgen import dbutil
from clgen import explore
from clgen import fetch
from clgen import log
from clgen import preprocess
from clgen.cache import Cache
from clgen.train import train


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


def atomize(corpus, vocab="char"):
    """
    Extract vocabulary of a corpus.

    Arguments:
        corpus (str): Corpus.
        voca (str, optional): Vocabularly type.

    Returns:
        list of str: Vocabularly.
    """
    def get_chars(corpus):
        counter = Counter(corpus)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        return chars

    def get_tokens(corpus):
        # FIXME: Actual token stream, not just whitespace separated.
        tokens = sorted(list(set(corpus.split())))
        return tokens

    # dispatch atomizer based on vocab type
    atomizers = {
        "char": get_chars,
        "tokens": get_tokens,
    }
    atomizer = atomizers.get(vocab, None)
    if atomizer is None:
        raise clgen.UserError(
            "Unknown vocabulary type '{}'. Supported types: {}".format(
                vocab, ", ".join(sorted(atomizers.keys()))))
    else:
        return atomizer(corpus)


class Corpus(clgen.CLgenObject):
    """
    Representation of a training corpus.
    """
    def __init__(self, path, isgithub=False, batch_size=50, seq_length=50):
        """
        Instantiate a corpus.

        If this is a new corpus, a number of files will be created, which may
        take some time.

        Arguments:
            path (str): Path to corpus.
            isgithub (bool): Whether corpus is from GitHub.
            batch_size (int): Batch size.
            seq_length (int): Sequence length.
        """
        path = unpack_directory_if_needed(fs.abspath(path))

        if not fs.isdir(path):
            raise clgen.UserError("Corpus path '{}' is not a directory"
                                  .format(path))

        self.hash = dirhash(path, 'sha1')

        self.isgithub = isgithub
        self.batch_size = batch_size
        self.seq_length = seq_length

        log.debug("corpus {hash}".format(hash=self.hash))

        self.cache = Cache(fs.path("corpus", self.hash))

        # TODO: Wrap file creation in try blocks, if any stage fails, delete
        # generated fail (if any)

        # create corpus database if not exists
        if not self.cache["kernels.db"]:
            self._create_kernels_db(path)

        # create corpus text if not exists
        if not self.cache["corpus.txt"]:
            self._create_txt()

        # preprocess if needed
        if self.cache["tensor.npy"] and self.cache["vocab.pkl"]:
            self._load_preprocessed()
        else:
            self._preprocess()

        self._create_batches()
        self.reset_batch_pointer()

    def _create_kernels_db(self, path):
        """creates and caches kernels.db"""
        log.debug("creating database")

        # create a database and put it in the cache
        tmppath = fs.path(self.cache.path, "kernels.db.tmp")
        dbutil.create_db(tmppath, github=self.isgithub)
        self.cache["kernels.db"] = tmppath

        # get a list of files in the corpus
        filelist = [f for f in fs.ls(path, abspaths=True, recursive=True)
                    if fs.isfile(f)]

        # import files into database
        fetch.fetch_fs(self.cache["kernels.db"], filelist)

        # preprocess files
        preprocess.preprocess_db(self.cache["kernels.db"])

        # print database stats
        if self.isgithub:
            explore.explore_gh(self.cache["kernels.db"])
        else:
            explore.explore(self.cache["kernels.db"])

    def _create_txt(self):
        """creates and caches corpus.txt"""
        log.debug("creating corpus")

        # TODO: additional options in corpus JSON to accomodate for EOF,
        # different encodings etc.
        tmppath = fs.path(self.cache.path, "corpus.txt.tmp")
        train(self.cache["kernels.db"], tmppath)
        self.cache["corpus.txt"] = tmppath

    def _preprocess(self):
        """creates and caches two files: vocab.pkl and tensor.npy"""
        log.debug("creating vocab and tensor files")
        input_file = self.cache["corpus.txt"]
        tmp_vocab_file = fs.path(self.cache.path, "vocab.tmp.pkl")
        tmp_tensor_file = fs.path(self.cache.path, "tensor.tmp.npy")

        with codecs.open(input_file, "r", encoding="utf-8") as infile:
            data = infile.read()
        self.atoms = atomize(data, vocab="char")

        self.vocab_size = len(self.atoms)
        self.vocab = dict(zip(self.atoms, range(len(self.atoms))))
        with open(tmp_vocab_file, 'wb') as f:
            cPickle.dump(self.atoms, f)
        # encode corpus with vocab
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tmp_tensor_file, self.tensor)

        self.cache["vocab.pkl"] = tmp_vocab_file
        self.cache["tensor.npy"] = tmp_tensor_file

    def _load_preprocessed(self):
        with open(self.cache["vocab.pkl"], 'rb') as infile:
            self.atoms = cPickle.load(infile)
        self.vocab_size = len(self.atoms)
        self.vocab = dict(zip(self.atoms, range(len(self.atoms))))
        self.tensor = np.load(self.cache["tensor.npy"])

    def _create_batches(self):
        log.debug("creating batches")
        self.num_batches = int(
            self.tensor.size / (self.batch_size * self.seq_length))

        if self.num_batches == 0:
            raise clgen.UserError(
                "Not enough data. Make seq_length and batch_size smaller.")

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def reset_batch_pointer(self):
        """
        Resets batch pointer to first batch.
        """
        self.pointer = 0

    def next_batch(self):
        """
        Fetch next batch indices.

        Returns:
            (np.array, np.array): X, Y batch tuple.
        """
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def set_batch_pointer(self, pointer):
        """
        Set batch pointer.

        Arguments:
            pointer (int): New batch pointer.
        """
        self.pointer = pointer

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
        log.debug("corpus from json")

        path = corpus_json.get("path", None)
        if path is None:
            raise clgen.UserError("no path found for corpus")

        isgithub = corpus_json.get("github", False)
        batch_size = corpus_json.get("batch_size", 50)
        seq_length = corpus_json.get("seq_length", 50)

        return Corpus(path=path, isgithub=isgithub, batch_size=batch_size,
                      seq_length=seq_length)
