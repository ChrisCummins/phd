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
"""Preprocess source code files for machine learning."""
import importlib
import pathlib
import typing
from importlib import util as importlib_util
from io import open

from datasets.github.scrape_repos.preprocessors import secrets
from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import public
from labm8.py import app

FLAGS = app.FLAGS

# Import type alias to public module.
PreprocessorFunction = public.PreprocessorFunction


def _ImportPreprocessorFromFile(module_path: pathlib.Path, function_name: str):
  """Import module from an absolute path to file, e.g. '/foo/bar.py'."""
  if not module_path.is_file():
    raise errors.UserError(f"File not found: {module_path}")
  try:
    spec = importlib_util.spec_from_file_location("module", str(module_path))
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
  except ImportError as e:
    raise errors.UserError(f'Failed to import module {module_path}: {e}')
  if not hasattr(module, function_name):
    raise errors.UserError(
        f'Function {function_name} not found in module {module_path}')
  return getattr(module, function_name)


def _ImportPreprocessorFromModule(module_name: str, function_name: str):
  """Import module from a fully qualified module name, e.g. 'foo.bar'."""
  try:
    module = importlib.import_module(module_name)
  except (ModuleNotFoundError, AttributeError):
    raise errors.UserError(f'Module {module_name} not found.')
  if not hasattr(module, function_name):
    raise errors.UserError(
        f'Function {function_name} not found in module {module_name}')
  function_ = getattr(module, function_name)
  if not function_.__dict__.get('is_clgen_preprocessor'):
    raise errors.UserError(
        f'Preprocessor {function_name} not decorated with @clgen_preprocessor')
  return function_


def GetPreprocessorFunction(name: str) -> public.PreprocessorFunction:
  """Lookup a preprocess function by name.

  A preprocessor is a function which takes a single argument 'text' of type str,
  and returns a str. The name is in the form <module>:<name>, where <name> is
  the name of a python function, and <module> is either a fully qualified module
  name, or an absolute path to the module file. For example, the name
  'deeplearning.clgen.preprocessors.cxx:Compile' will return the function
  'Compile' in the module 'deeplearning.clgen.preprocessors.cxx'. The name
  '/tmp/my_preprocessors.py:Transform' will return the function Transform() in
  the module defined at '/tmp/my_preprocessors.py'.

  Args:
    name: The name of the preprocessor to get.

  Returns:
    The python preprocessor function.

  Raises:
    UserError: If the requested name cannot be found or is not a
      @clgen_preprocessor decorated function.
  """
  components = name.split(':')
  if len(components) != 2:
    raise errors.UserError(f'Invalid preprocessor name {name}')
  module_name, function_name = components
  if module_name[0] == '/':
    return _ImportPreprocessorFromFile(pathlib.Path(module_name), function_name)
  else:
    return _ImportPreprocessorFromModule(module_name, function_name)


def Preprocess(text: str, preprocessors: typing.List[str]) -> str:
  """Preprocess a text using the given preprocessor pipeline.

  If preprocessing succeeds, the preprocessed text is returned. If preprocessing
  fails (in an expected way, for example by trying to compile incorrect code),
  a BadCodeException is raised. Any other error leads to an InternalError.


  Args:
    text: The input to be preprocessed.
    preprocessors: The list of preprocessor functions to run. These will be
      passed to GetPreprocessorFunction() to resolve the python implementations.

  Returns:
    Preprocessed source input as a string.

  Raises:
    UserError: If the requested preprocessors cannot be loaded.
    BadCodeException: If one of the preprocessors rejects the input.
    InternalException: In case of some other error.
  """
  preprocessor_functions = [GetPreprocessorFunction(p) for p in preprocessors]
  for preprocessor in preprocessor_functions:
    text = preprocessor(text)
  return text


def PreprocessFile(path: str, preprocessors: typing.List[str],
                   inplace: bool) -> str:
  """Preprocess a file and optionally update it.

  Args:
    text: The input to be preprocessed.
    preprocessors: The list of preprocessor functions to run. These will be
      passed to GetPreprocessorFunction() to resolve the python implementations.
    inplace: If True, the input file is overwritten with the preprocessed code,
      unless the preprocessing fails. If the preprocessing fails, the input
      file is left unmodified.

  Returns:
    Preprocessed source input as a string.

  Raises:
    UserError: If the requested preprocessors cannot be loaded.
    BadCodeException: If one of the preprocessors rejects the input.
    InternalException: In case of some other error.
  """
  with open(path) as infile:
    contents = infile.read()
  preprocessed = Preprocess(contents, preprocessors)
  if inplace:
    with open(path, 'w') as outfile:
      outfile.write(preprocessed)
  return preprocessed


@public.clgen_preprocessor
def RejectSecrets(text: str) -> str:
  """Test for secrets such as private keys in a text.

  Args:
    text: The text to check.

  Returns:
    The unmodified text.

  Raises:
    BadCodeException: In case the text contains secrets.
  """
  try:
    secrets.ScanForSecrets(text)
    return text
  except secrets.TextContainsSecret as e:
    raise errors.BadCodeException(str(e))
