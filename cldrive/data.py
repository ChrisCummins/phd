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
    ARANGE = partial(np.arange)
    ZEROS = partial(np.zeros)
    ONES = partial(np.ones)

    def __call__(self, numpy_type: np.dtype, *args, **kwargs):
        """ generate arrays of data """
        return self.value(*args, **kwargs).astype(numpy_type)

    @staticmethod
    def from_str(string: str) -> 'Generator':
        if string == "rand":
            return Generator.RAND
        elif string == "arange":
            return Generator.ARANGE
        elif string == "zeros":
            return Generator.ZEROS
        elif string == "ones":
            return Generator.ONES
        else:
            raise InputTypeError


def make_data(src: str, size: int, data_generator: Generator,
              scalar_val: float=None) -> np.array:
    """
    Generate data for OpenCL kernels.

    Creates a numpy array for each OpenCL argument, except arguments with the
    'local' qualifier, since those are instantiated.

    Returns
    -------
    np.array
        The generated data.

    Raises
    ------
    TypeError
        If any of the input arguments are of incorrect type.
    ValueError
        If any of the arguments cannot be interpreted.

    Examples
    --------

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ZEROS)
    array([array([0, 0, 0], dtype=int32), array([3], dtype=int32)], dtype=object)

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ONES)
    array([array([1, 1, 1], dtype=int32), array([3], dtype=int32)], dtype=object)

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ARANGE)
    array([array([0, 1, 2], dtype=int32), array([3], dtype=int32)], dtype=object)

    Use `scalar_val` parameter to fix the value of scalar arguments:

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ARANGE, scalar_val=100)
    array([array([0, 1, 2], dtype=int32), array([100], dtype=int32)], dtype=object)
    """
    # check the input types
    assert_or_raise(isinstance(src, str), TypeError)
    assert_or_raise(isinstance(data_generator, Generator), TypeError,
                    "invalid argument type for enum data_generator")

    if scalar_val is None:
        scalar_val = size

    data = []
    for arg in extract_args(src):
        if arg.address_space == "global" or arg.address_space == "constant":
            argdata = data_generator(arg.numpy_type, size * arg.vector_width)
        elif arg.address_space == "local":
            # we don't need to generate data for local memory
            continue
        elif not arg.is_pointer:
            # scalar values are still arrays, so e.g. 'float4' is an array of
            # 4 floats. Each component of a scalar value is the flattened
            # global size, e.g. with gsize (32,2,1), scalar arugments have the
            # value 32 * 2 * 1 = 64.
            scalar_val = [scalar_val] * arg.vector_width
            argdata = np.array(scalar_val).astype(arg.numpy_type)
        else:
            # argument is neither global or local, but is a pointer?
            raise ValueError(f"unknown argument type '{arg}'")

        data.append(argdata)

    return np.array(data)


def zeros(*args, **kwargs) -> np.array:
    return make_data(*args, data_generator=Generator.ZEROS, **kwargs)


def ones(*args, **kwargs) -> np.array:
    return make_data(*args, data_generator=Generator.ONES, **kwargs)


def arange(*args, **kwargs) -> np.array:
    return make_data(*args, data_generator=Generator.ARANGE, **kwargs)


def rand(*args, **kwargs) -> np.array:
    return make_data(*args, data_generator=Generator.RAND, **kwargs)
