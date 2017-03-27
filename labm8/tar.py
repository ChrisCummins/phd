# Copyright (C) 2015, 2016 Chris Cummins.
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
Tarball util.
"""

import tarfile

from labm8 import fs


def unpack_archive(*components, **kwargs) -> str:
    """
    Unpack a compressed archive.

    Arguments:
        *components (str[]): Absolute path.
        **kwargs (dict, optional): Set "compression" to compression type.
            Default: bz2. Set "dir" to destination directory. Defaults to the
            directory of the archive.

    Returns:
        str: Path to directory.
    """
    path = fs.abspath(*components)
    compression = kwargs.get("compression", "bz2")
    dir = kwargs.get("dir", fs.dirname(path))

    fs.cd(dir)
    tar = tarfile.open(path, "r:" + compression)
    tar.extractall()
    tar.close()
    fs.cdpop()

    return dir
