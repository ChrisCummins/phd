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
from enum import Enum
from functools import partial

import numpy as np


from cldrive import *


class Generator(Enum):
    # We wrap functions in a partial so that they are interpreted as attributes
    # rather than methods. See: http://stackoverflow.com/a/40339397
    RAND = partial(np.random.rand)
    SEQ = partial(np.arange)
    ZEROS = partial(np.zeros)
    ONES = partial(np.ones)

    def __call__(self, numpy_type: np.dtype, *args, **kwargs):
        """ generate arrays of data """
        return self.value(*args, **kwargs).astype(numpy_type)

    @staticmethod
    def from_str(string: str) -> 'Generator':
        if string == "rand":
            return Generator.RAND
        elif string == "seq":
            return Generator.SEQ
        elif string == "zeros":
            return Generator.ZEROS
        elif string == "ones":
            return Generator.ONES
        else:
            raise InputTypeError


def make_data(drive: Driver, size: int, data_generator: Generator,
              scalar_val: float=None) -> np.array:
    """
    Generate data for OpenCL kernels.

    Creates a numpy array for each OpenCL argument, except arguments with the
    'local' qualifier, since those are instantiated.

    Returns:
        np.array: The generated data.

    Raises:
        InputTypeError: If any of the input arguments are of incorrect type.
    """
    # check the input types
    _assert_or_raise(isinstance(driver, Driver), TypeError)
    _assert_or_raise(isinstance(data_generator, Generator), TypeError)

    if scalar_val is None:
        scalar_val = size

    args_with_inputs = [arg for arg in args if arg.has_input]

    data = []
    for arg in args_with_inputs:
        if arg.is_global:
            argdata = data_generator(arg.numpy_type, size * arg.vector_width)
        else:
            # scalar values are still arrays, so e.g. 'float4' is an array of
            # 4 floats. Each component of a scalar value is the flattened
            # global size, e.g. with gsize (32,2,1), scalar arugments have the
            # value 32 * 2 * 1 = 64.
            scalar_val = [scalar_val] * arg.vector_width
            argdata = np.array(scalar_val).astype(arg.numpy_type)

        data.append(argdata)

    return np.array(data)