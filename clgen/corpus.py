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
from clgen.cache import Cache


class CorpusCache(Cache):
    def __init__(self):
        super(CorpusCache, self).__init__("corpus")


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

        cache = CorpusCache()
        self.dbname = self.hash + ".db"

        # create corpus database if not exists
        if not cache[self.dbname]:
            self._create_db(path)

    def _create_db(self, path):
        cache = CorpusCache()

        # create a database and put it in the cache
        tmppath = ".corpus.tmp.db"
        dbutil.create_db(".corpus.tmp.db", github=self.isgithub)
        cache[self.dbname] = ".corpus.tmp.db"

        # get a list of files in the corpus
        filelist = [f for f in fs.ls(path, abspaths=True, recursive=True)
                    if fs.isfile(f)]

        # import files into database
        fetch.fs(cache[self.dbname], filelist)

        # preprocess files
        preprocess.preprocess_db(cache[self.dbname])

        # print database stats
        if self.isgithub:
            explore.explore_gh(cache[self.dbname])
        else:
            explore.explore(cache[self.dbname])

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
