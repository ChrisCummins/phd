import functools
from enum import Enum

import numpy as np

from gpu.cldrive import args as _args
from lib.labm8 import err


class Generator(Enum):
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
      raise TypeError


def make_data(src: str, size: int, data_generator: Generator,
              scalar_val: float = None) -> np.array:
  """
  Generate data for OpenCL kernels.

  Creates a numpy array for each OpenCL argument, except arguments with the
  'local' qualifier, since those are instantiated.

  Returns:
    The generated data as an np.array.

  Raises:
    TypeError: If any of the input arguments are of incorrect type.
    ValueError: If any of the arguments cannot be interpreted.

  Examples:

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ZEROS)
    array([array([0, 0, 0], dtype=int32), array([3], dtype=int32)],
          dtype=object)

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ONES)
    array([array([1, 1, 1], dtype=int32), array([3], dtype=int32)],
          dtype=object)

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ARANGE)
    array([array([0, 1, 2], dtype=int32), array([3], dtype=int32)],
          dtype=object)

    Use `scalar_val` parameter to fix the value of scalar arguments:

    >>> make_data("kernel void A(global int* a, const int b) {}", 3, Generator.ARANGE, scalar_val=100)
    array([array([0, 1, 2], dtype=int32), array([100], dtype=int32)],
          dtype=object)
  """
  # check the input types
  err.assert_or_raise(isinstance(src, str), TypeError)
  err.assert_or_raise(isinstance(data_generator, Generator), TypeError,
                      "invalid argument type for enum data_generator")

  if scalar_val is None:
    scalar_val = size

  data = []
  for arg in _args.extract_args(src):
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


def zeros(*args, **kwargs) -> np.array:
  return make_data(*args, data_generator=Generator.ZEROS, **kwargs)


def ones(*args, **kwargs) -> np.array:
  return make_data(*args, data_generator=Generator.ONES, **kwargs)


def arange(*args, **kwargs) -> np.array:
  return make_data(*args, data_generator=Generator.ARANGE, **kwargs)


def rand(*args, **kwargs) -> np.array:
  return make_data(*args, data_generator=Generator.RAND, **kwargs)
