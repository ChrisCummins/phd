#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Programming language.

Attributes:
    __languages__ (Dict[str, Language]): List of all available languages.
"""

class Generator(object):
    @property
    def num_programs(self) -> int:
        raise NotImplementedError("abstract class")


class Language(object):
    """
    Abstract interface for a programming language.

    Attributes:
        __generators__ (List[Generator]): List of available generators.
    """
    def __init__(self):
        raise NotImplementedError("abstract class")

    def mkgenerator(self, name: str) -> Generator:
        raise NotImplementedError("abstract class")

# Deferred importing of languages, since the modules may need to import this
# file.
from dsmith.langs.opencl import OpenCL

__languages__ = {
    "opencl": OpenCL,
}

def mklang(name: str) -> Language:
    """

    """
    lang = __languages__.get(name)
    if not lang:
        raise ValueError("Unknown language")
    return lang()
