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
CLgen persistent cache mechanism.
"""
import re

from labm8 import fs
from shutil import move
from six import string_types

import clgen
from clgen import log

ROOT = fs.path("~", ".cache", "clgen", clgen.version())


class Cache404(clgen.File404):
    """
    Error thrown for cache misses.
    """
    pass


class Cache(clgen.CLgenObject):
    """
    Persistent filesystem cache.
    """
    def __init__(self, name):
        """
        Create filesystem cache.
        """
        self.path = fs.path(ROOT, name)
        self.name = name

        fs.mkdir(self.path)

    def empty(self):
        """
        Empty the filesystem cache.
        """
        log.debug("empty cache {path}".format(path=self.path))
        fs.rm(self.path)

    def mapkey(self, key):
        """
        Map a key to an internal key.

        Key mapping in case the keys stored in the cache should be distinct
        from the keys used to access the cache.

        Arguments:
            key (str): Key.

        Returns:
            str: Mapped key.
        """
        return key

    def escape(self, key):
        """
        Escape key to path.

        Arguments:
            key (str): Key.

        Returns:
            str: Escaped key.
        """
        return re.sub(r'[ \\/]+', '_', key)

    def keypath(self, key):
        """
        Return path to key in cache.

        Arguments:
            key (str): Key.

        Returns:
            str: Absolute path.
        """
        return fs.path(self.path, self.escape(self.mapkey(key)))

    def _incache(self, path):
        """
        Assert that file is in cache.

        Arguments:
            path (str): File path.

        Returns:
            str: File path.

        Raises:
            Cache404: If file does not exist.
        """
        if not fs.exists(path):
            raise Cache404("file '{path}' not found".format(path=path))
        return path

    def __getitem__(self, key):
        """
        Get path to file in cache.

        Arguments:
            key (str): Key.

        Returns:
            str: Path to cache value, or bool False if not found.
        """
        assert(isinstance(key, string_types))

        try:
            return self._incache(self.keypath(key))
        except Cache404:
            return False

    def __setitem__(self, key, value):
        """
        Emplace file in cache.

        Arguments:
            key (str): Key.
            value (str): Path of file to insert in cache.

        Raises:
            clgen.File404: If no "value" does nto exist.
        """
        assert(isinstance(key, string_types))
        assert(isinstance(value, string_types))

        clgen.must_exist(value, error=clgen.File404)

        path = self.keypath(key)
        move(value, path)
        log.debug("cached {path}"
                  .format(key=key, path=path))

    def __delitem__(self, key):
        """
        Delete cached file.

        Arguments:
            key (str): Key.

        Raises:
            Cache404: If file not in cache.
        """
        assert(isinstance(key, string_types))

        path = self._incache(self.keypath(key))
        fs.rm(path)
