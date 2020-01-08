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
"""Unit tests for //deeplearning/clgen/preprocessors/preprocessors.py."""
import pathlib

import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.preprocessors import public
from labm8.py import app
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS


@public.clgen_preprocessor
def MockPreprocessor(text: str) -> str:
  """A mock preprocessor."""
  del text
  return "PREPROCESSED"


@public.clgen_preprocessor
def MockPreprocessorBadCode(text: str) -> str:
  """A mock preprocessor which raises a BadCodeException."""
  del text
  raise errors.BadCodeException("bad code")


@public.clgen_preprocessor
def MockPreprocessorInternalError(text: str) -> str:
  """A mock preprocessor which raises a BadCodeException."""
  del text
  raise errors.InternalError("internal error")


def MockUndecoratedPreprocessor(text: str) -> str:
  """A mock preprocessor which is not decorated with @clgen_preprocessor."""
  return text


# GetPreprocessFunction() tests.


def test_GetPreprocessFunction_empty_string():
  """Test that an UserError is raised if no preprocessor is given."""
  with test.Raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction("")
  assert "Invalid preprocessor name" in str(e_info.value)


def test_GetPreprocessFunction_missing_module():
  """Test that UserError is raised if module not found."""
  with test.Raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction("not.a.real.module:Foo")
  assert "not found" in str(e_info.value)


def test_GetPreprocessFunction_missing_function():
  """Test that UserError is raised if module exists but function doesn't."""
  with test.Raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction(
      "deeplearning.clgen.preprocessors.preprocessors_test:Foo"
    )
  assert "not found" in str(e_info.value)


def test_GetPreprocessFunction_undecorated_preprocessor():
  """Test that an UserError is raised if preprocessor not decorated."""
  with test.Raises(errors.UserError) as e_info:
    preprocessors.GetPreprocessorFunction(
      "deeplearning.clgen.preprocessors.preprocessors_test"
      ":MockUndecoratedPreprocessor"
    )
  assert "@clgen_preprocessor" in str(e_info.value)


def test_GetPreprocessFunction_mock_preprocessor():
  """Test that a mock preprocessor can be found."""
  f = preprocessors.GetPreprocessorFunction(
    "deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor"
  )
  assert f.__name__ == "MockPreprocessor"


def test_GetPreprocessorFunction_absolute_path(tempdir: pathlib.Path):
  """Test loading module from absolute path to file."""
  path = tempdir / "preprocessor.py"
  fs.Write(
    path,
    """
def Preprocess(src: str) -> str:
  return src.replace('a', 'b')
""".encode(
      "utf-8"
    ),
  )

  f = preprocessors.GetPreprocessorFunction(f"{path}:Preprocess")
  assert f("abc") == "bbc"


def test_GetPreprocessorFunction_absolute_path_with_dep(tempdir: pathlib.Path):
  """Test loading module from file which has a dependency."""
  lib_module = tempdir / "lib_module.py"
  fs.Write(
    lib_module,
    """
def PreprocessImplementation(src):
  return src.replace('b', 'c')
""".encode(
      "utf-8"
    ),
  )

  path = tempdir / "lib_module.py"
  fs.Write(
    path,
    """
from . import lib_module
def Preprocess(src):
  return lib_module.PreprocessImplementation(src)
""".encode(
      "utf-8"
    ),
  )

  with test.Raises(errors.UserError):
    preprocessors.GetPreprocessorFunction(f"{path}:Preprocess")


def test_GetPreprocessorFunction_absolute_path_not_found(tempdir: pathlib.Path):
  """Test loading module when file not found."""
  path = tempdir / "foo.py"
  fs.Write(path, "".encode("utf-8"))
  with test.Raises(errors.UserError):
    preprocessors.GetPreprocessorFunction(f"{path}:NotFound")


def test_GetPreprocessorFunction_absolute_function_not_found(
  tempdir: pathlib.Path,
):
  """Test loading module when file not found."""
  with test.Raises(errors.UserError):
    preprocessors.GetPreprocessorFunction(f"{tempdir}/foo.py:Preprocess")


# Preprocess() tests.


def test_Preprocess_no_preprocessors():
  """Test unmodified output if no preprocessors."""
  assert preprocessors.Preprocess("hello", []) == "hello"


def test_Preprocess_mock_preprocessor():
  """Test unmodified output if no preprocessors."""
  assert (
    preprocessors.Preprocess(
      "hello",
      ["deeplearning.clgen.preprocessors.preprocessors_test:MockPreprocessor"],
    )
    == "PREPROCESSED"
  )


def test_Preprocess_mock_preprocessor_bad_code():
  """Test that BadCodeException is propagated."""
  with test.Raises(errors.BadCodeException):
    preprocessors.Preprocess(
      "",
      [
        "deeplearning.clgen.preprocessors.preprocessors_test"
        ":MockPreprocessorBadCode"
      ],
    )


def test_Preprocess_mock_preprocessor_internal_error():
  """Test that InternalError is propagated."""
  with test.Raises(errors.InternalError):
    preprocessors.Preprocess(
      "",
      [
        "deeplearning.clgen.preprocessors.preprocessors_test"
        ":MockPreprocessorInternalError"
      ],
    )


# RejectSecrets() tests.


def test_Preprocess_RejectSecrets():
  """Test that InternalError is propagated."""
  assert (
    preprocessors.Preprocess(
      "Hello, world!",
      ["deeplearning.clgen.preprocessors.preprocessors" ":RejectSecrets"],
    )
    == "Hello, world!"
  )


def test_Preprocess_RejectSecrets():
  """Test that InternalError is propagated."""
  with test.Raises(errors.BadCodeException):
    preprocessors.Preprocess(
      "-----BEGIN RSA PRIVATE KEY-----",
      ["deeplearning.clgen.preprocessors.preprocessors" ":RejectSecrets"],
    )


# Benchmarks.


def test_benchmark_GetPreprocessFunction_mock(benchmark):
  """Benchmark GetPreprocessFunction."""
  benchmark(
    preprocessors.GetPreprocessorFunction,
    "deeplearning.clgen.preprocessors.preprocessors_test" ":MockPreprocessor",
  )


if __name__ == "__main__":
  test.Main()
