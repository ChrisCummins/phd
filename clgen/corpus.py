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
from clgen import log


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
        clgen.unpack_archive(path)
        return re.sub(r'.tar.bz2$', '', path)

    if fs.isfile(path + ".tar.bz2"):
        clgen.unpack_archive(path + ".tar.bz2")
        return path

    return path


class Corpus:
    """
    Representation of a training corpus.
    """
    def __init__(self, path):
        path = fs.abspath(path)

        path = unpack_directory_if_needed(path)

        if not fs.isdir(path):
            raise clgen.UserError("Corpus '{}' must be a directory"
                                  .format(path))

        self.hash = dirhash(path, 'sha1')

        log.debug("Corpus {hash} initialized".format(hash=self.hash))
