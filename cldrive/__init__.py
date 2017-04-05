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
'''
Run arbitrary OpenCL kernels.
'''
from collections import namedtuple
from enum import Enum, auto

import numpy as np


NDRange = namedtuple('NDRange', ['x', 'y', 'z'])


class Inputs(Enum):
    RAND = auto()
    SEQ = auto()
    ZEROS = auto()
    ONES = auto()


def run(src: str, gsize: NDRange, lsize: NDRange,
        inputs: Inputs=Inputs.SEQ) -> np.array:
    assert(isinstance(src, str))
    assert(isinstance(gsize, NDRange))
    assert(isinstance(lsize, NDRange))
    assert(isinstance(inputs, Inputs))

    # WIP
    return np.array([[0, 2, 4, 6]])