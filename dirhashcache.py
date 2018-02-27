# Copyright (C) 2015-2017 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
#
"""
The fast hash cache.

Checksum directories and cache results. If a directory has not been modified,
subsequent hashes are cache hits. Hashes are recomputed lazily, when a
directory (or any of its subdirectories) have been modified.
"""
import checksumdir
import os
import sqlite3
import time

from labm8 import fs


class DirHashCache(object):
    def __init__(self, path, hash='sha1'):
        """
        Instantiate a directory checksum cache.

        Arguments:
            path (str): Path to persistent cache store.
            hash (str, optional): Hash algorithm to use, e.g. 'md5', 'sha1'.
        """
        self.path = fs.path(path)
        self.hash = hash

        db = sqlite3.connect(self.path)
        c = db.cursor()
        c.execute("""\
CREATE TABLE IF NOT EXISTS dirhashcache (
        path TEXT NOT NULL,
        date DATETIME NOT NULL,
        hash TEXT NOT NULL,
        PRIMARY KEY(path)
)""")
        db.commit()
        db.close()

    def clear(self):
        """
        Remove all cache entries.
        """
        db = sqlite3.connect(self.path)
        c = db.cursor()
        c.execute("DELETE FROM dirhashcache")
        db.commit()
        db.close()

    def dirhash(self, path, **dirhash_opts):
        """
        Compute the hash of a directory.

        Arguments:
           path: Directory.
           **dirhash_opts: Additional options to checksumdir.dirhash().

        Returns:
            str: Checksum of directory.
        """
        path = fs.path(path)
        last_modified = time.ctime(max(
            max(os.path.getmtime(os.path.join(root, file)) for file in files)
            for root,_,files in os.walk(path)))

        db = sqlite3.connect(self.path)
        c = db.cursor()
        c.execute("SELECT date, hash FROM dirhashcache WHERE path=?", (path,))
        cached = c.fetchone()

        if cached:
            cached_date, cached_hash = cached
            if cached_date == last_modified:
                # cache hit
                dirhash = cached_hash
            else:
                # out of date cache
                dirhash = checksumdir.dirhash(path, self.hash, **dirhash_opts)
                c.execute("UPDATE dirhashcache SET date=?, hash=? WHERE path=?",
                          (last_modified, dirhash, path))
                db.commit()
        else:
            # new entry
            dirhash = checksumdir.dirhash(path, self.hash, **dirhash_opts)
            c.execute("INSERT INTO dirhashcache VALUES (?,?,?)",
                      (path, last_modified, dirhash))
            db.commit()

        db.close()
        return dirhash
