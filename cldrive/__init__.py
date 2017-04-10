# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Run arbitrary OpenCL kernels.

Attributes
----------
__version__ : str
    PEP 440 compiliant package version.
"""
from pkg_resources import require


__version__: str = require("cldrive")[0].version


def assert_or_raise(stmt: bool, exception: Exception,
                    *exception_args, **exception_kwargs) -> None:
    """
    If the statement is false, raise the given exception.
    """
    if not stmt:
        raise exception(*exception_args, **exception_kwargs)

# note to future me: the order of imports here is important.
from cldrive.env import *
from cldrive.args import *
from cldrive.driver import *
from cldrive.data import *
