# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""Generate data for OpenCL arguments."""
import functools
from enum import Enum

import numpy as np

from gpu.cldrive.legacy import args as _args
from labm8.py import app


class Generator(Enum):
  """Data generator types."""
  # We wrap functions in a partial so that they are interpreted as attributes
  # rather than methods. See: http://stackoverflow.com/a/40339397
  RAND = functools.partial(np.random.rand)
  ARANGE = functools.partial(lambda x: np.arange(0, x))
  ZEROS = functools.partial(np.zeros)
  ONES = functools.partial(np.ones)

  def __call__(self, numpy_type: np.dtype, *args, **kwargs):
    """ generate arrays of data """
    return self.value(*args, **kwargs).astype(numpy_type)

  @staticmethod
  def FromString(string: str) -> 'Generator':
    if string == "rand":
      return Generator.RAND
    elif string == "arange":
      return Generator.ARANGE
    elif string == "zeros":
      return Generator.ZEROS
    elif string == "ones":
      return Generator.ONES
    else:
      raise ValueError(f"Unknown generator name: '{string}'")


def MakeData(src: str,
             size: int,
             data_generator: Generator,
             scalar_val: float = None) -> np.array:
  """Generate data for OpenCL kernels.

  Creates a numpy array for each OpenCL argument, except arguments with the
  'local' qualifier, since those are instantiated.

  Returns:
    The generated data as an np.array.

  Raises:
    TypeError: If any of the input arguments are of incorrect type.
    ValueError: If any of the arguments cannot be interpreted.

  Examples:
    >>> MakeData("kernel void A(global int* a, const int b) {}", 3, Generator.ZEROS)
    array([array([0, 0, 0], dtype=int32), array([3], dtype=int32)],
          dtype=object)

    >>> MakeData("kernel void A(global int* a, const int b) {}", 3, Generator.ONES)
    array([array([1, 1, 1], dtype=int32), array([3], dtype=int32)],
          dtype=object)

    >>> MakeData("kernel void A(global int* a, const int b) {}", 3, Generator.ARANGE)
    array([array([0, 1, 2], dtype=int32), array([3], dtype=int32)],
          dtype=object)

    Use `scalar_val` parameter to fix the value of scalar arguments:

    >>> MakeData("kernel void A(global int* a, const int b) {}", 3, Generator.ARANGE, scalar_val=100)
    array([array([0, 1, 2], dtype=int32), array([100], dtype=int32)],
          dtype=object)
  """
  # check the input types
  app.AssertOrRaise(isinstance(src, str), TypeError)
  app.AssertOrRaise(isinstance(data_generator, Generator), TypeError,
                    "invalid argument type for enum data_generator")

  if scalar_val is None:
    scalar_val = size

  data = []
  for arg in _args.GetKernelArguments(src):
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
      argdata = np.array([scalar_val] * arg.vector_width).astype(arg.numpy_type)
    else:
      # argument is neither global or local, but is a pointer?
      raise ValueError(f"unknown argument type '{arg}'")

    data.append(argdata)

  return np.array(data)


def MakeZeros(*args, **kwargs) -> np.array:
  """Make zero-valued arguments."""
  return MakeData(*args, data_generator=Generator.ZEROS, **kwargs)


def MakeOnes(*args, **kwargs) -> np.array:
  """Make one-valued arguments."""
  return MakeData(*args, data_generator=Generator.ONES, **kwargs)


def MakeArange(*args, **kwargs) -> np.array:
  """Make sequentially valued arguments."""
  return MakeData(*args, data_generator=Generator.ARANGE, **kwargs)


def MakeRand(*args, **kwargs) -> np.array:
  """Make arguments of random values."""
  return MakeData(*args, data_generator=Generator.RAND, **kwargs)
