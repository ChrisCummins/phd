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
Compiler fuzzing through deep learning.

Attributes:
    __version__ (str): PEP 440 compliant version string.
    version_info: Tuple with 'major', 'minor', 'micro', 'releaselevel'
        components.
"""
from collections import namedtuple
from pkg_resources import resource_filename, resource_string, require

from dsmith._config import *

__author__ = "Chris Cummins"
__copyright__ = "Copyright 2017 Chris Cummins"
__license__ = "GPL v3"
__version__ = require("dsmith")[0].version
__maintainer__ = __author__
__email__ = "chrisc.101@gmail.com"
__status__ = "Prototype"


# version_info tuple
_major = int(__version__.split(".")[0])
_minor = int(__version__.split('.')[1]) if len(__version__.split('.')) > 1 else 0
_micro = int(__version__.split('.')[2]) if len(__version__.split('.')) > 2 else 0
_releaselevel = __version__.split('.')[3] if len(__version__.split('.')) > 3 else 'final'

version_info_t = namedtuple('version_info_t', ['major', 'minor', 'micro', 'releaselevel'])
version_info = version_info_t(_major, _minor, _micro, _releaselevel)


class DSmithError(Exception):
    """
    Top level error. Never directly thrown.
    """
    pass


class InternalError(DSmithError):
    """
    An internal module error. This class of errors should not leak outside of
    the module into user code.
    """
    pass


class UserError(DSmithError):
    """
    Raised in case of bad user interaction, e.g. an invalid argument.
    """
    pass


class Filesystem404(InternalError):
    """
    Path not found.
    """
    pass
