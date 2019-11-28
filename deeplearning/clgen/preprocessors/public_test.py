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
"""Unit tests for //deeplearning/clgen/preprocessors/public.py."""
import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import public
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_clgen_preprocessor_good():
  """Test clgen_preprocessor decorator on a valid function."""

  @public.clgen_preprocessor
  def MockPreprocessor(text: str) -> str:
    """Mock preprocessor."""
    return text

  assert MockPreprocessor("foo") == "foo"


def test_clgen_preprocessor_missing_return_type():
  """Test clgen_preprocessor on a function missing a return type hint."""
  with test.Raises(errors.InternalError):

    @public.clgen_preprocessor
    def MockPreprocessor(test: str):
      """Mock preprocessor with a missing return type hint."""
      del test


def test_clgen_preprocessor_missing_argument_type():
  """Test clgen_preprocessor on a function missing an argument type hint."""
  with test.Raises(errors.InternalError):

    @public.clgen_preprocessor
    def MockPreprocessor(test) -> str:
      """Mock preprocessor with a missing argument type hint."""
      del test


def test_clgen_preprocessor_incorrect_argument_name():
  """Test clgen_preprocessor on a function missing an argument type hint."""
  with test.Raises(errors.InternalError):

    @public.clgen_preprocessor
    def MockPreprocessor(foo: str) -> str:
      """Mock preprocessor with a mis-named argument."""
      del foo


if __name__ == "__main__":
  test.Main()
