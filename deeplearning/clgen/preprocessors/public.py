# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file defines the decorator for marking a CLgen preprocessor function."""
import typing

from absl import flags

from deeplearning.clgen import errors


FLAGS = flags.FLAGS

# Type hint for a preprocessor function. See @clgen_preprocess for details.
PreprocessorFunction = typing.Callable[[str], str]


def clgen_preprocessor(func: PreprocessorFunction) -> PreprocessorFunction:
  """A decorator which marks a function as a CLgen preprocessor.

  A CLgen preprocessor is accessible using GetPreprocessFunction(), and is a
  function which accepts a single parameter 'text', and returns a string.
  Type hinting is used to ensure that any function wrapped with this decorator
  has the appropriate argument and return type. If the function does not, an
  InternalError is raised at the time that the module containing the function
  is imported.

  Args:
    func: The preprocessor function to decorate.

  Returns:
    The decorated preprocessor function.

  Raises:
    InternalError: If the function being wrapped does not have the signature
      'def func(text: str) -> str:'.
  """
  type_hints = typing.get_type_hints(func)
  if not type_hints == {'text': str, 'return': str}:
    raise errors.InternalError(
        f'Preprocessor {func.__name__} does not have signature '
        f'"def {func.__name__}(text: str) -> str".')
  func.__dict__['is_clgen_preprocessor'] = True
  return func
