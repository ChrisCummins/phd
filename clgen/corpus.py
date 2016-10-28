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

from checksumdir import dirhash
from labm8 import fs

import clgen
from clgen import dbutil
from clgen import explore
from clgen import fetch
from clgen import log
from clgen import preprocess
from clgen import torch_rnn
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


class Corpus:
    """
    Representation of a training corpus.
    """
    def __init__(self, path, isgithub=False):
        """
        Instantiate a corpus.

        If this is a new corpus, a database will be created for it
        """
        path = fs.abspath(path)

        path = unpack_directory_if_needed(path)

        if not fs.isdir(path):
            raise clgen.UserError("Corpus '{}' must be a directory"
                                  .format(path))

        self.hash = dirhash(path, 'sha1')
        self.isgithub = isgithub

        log.debug("corpus {hash} initialized".format(hash=self.hash))

        cache = Cache(self.hash)

        # TODO: Wrap file creation in try blocks, if any stage fails, delete
        # generated fail (if any)

        # create corpus database if not exists
        if not cache["kernels.db"]:
            self._create_kernels_db(path)

        # create corpus text if not exists
        if not cache["corpus.txt"]:
            self._create_txt()

        # create LSTM training files if not exists
        if not cache["corpus.json"] or not cache["corpus.h5"]:
            self._lstm_preprocess()

    def _create_kernels_db(self, path):
        cache = Cache(self.hash)

        log.debug("creating database...")

        # create a database and put it in the cache
        tmppath = fs.path(cache.path, "kernels.db.tmp")
        dbutil.create_db(tmppath, github=self.isgithub)
        cache["kernels.db"] = tmppath

        # get a list of files in the corpus
        filelist = [f for f in fs.ls(path, abspaths=True, recursive=True)
                    if fs.isfile(f)]

        # import files into database
        fetch.fs(cache["kernels.db"], filelist)

        # preprocess files
        preprocess.preprocess_db(cache["kernels.db"])

        # print database stats
        if self.isgithub:
            explore.explore_gh(cache["kernels.db"])
        else:
            explore.explore(cache["kernels.db"])

    def _create_txt(self):
        cache = Cache(self.hash)

        log.debug("creating corpus...")

        # TODO: additional options in corpus JSON to accomodate for EOF,
        # different encodings etc.
        tmppath = fs.path(cache.path, "corpus.txt.tmp")
        train(cache["corpus.txt"], tmppath)
        cache["corpus.txt"] = tmppath

    def _lstm_preprocess(self):
        cache = Cache(self.hash)

        log.debug("creating training set...")
        tmppaths = (fs.path(cache.path, "corpus.json.tmp"),
                    fs.path(cache.path, "corpus.h5.tmp"))
        torch_rnn.preprocess(cache["corpus.txt"], *tmppaths)
        cache["corpus.json"] = tmppaths[0]
        cache["corpus.h5"] = tmppaths[1]

    @staticmethod
    def from_json(corpus_json):
        """
        Instantiate Corpus from JSON.
        """
        log.debug("reading corpus json...")

        path = corpus_json.get("path", None)
        if path is None:
            raise clgen.UserError("key 'path' not in corpus JSON")
        isgithub = corpus_json.get("github", False)

        return Corpus(path, isgithub=isgithub)
