# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocess files within a dataset."""
import importlib

import pathlib
import typing

from datasets.github.scrape_repos.preprocessors import public
from labm8 import app

FLAGS = app.FLAGS

# Import type alias so that users don't have to import public.
PreprocessorFunction = public.PreprocessorFunction


def GetPreprocessorFunction(name: str) -> public.PreprocessorFunction:
  """Lookup a dataset preprocess function by name.

  A dataset preprocessor is a function which two arguments: a root directory,
  and a relative path to a file within that directory. The name is the fully
  qualified name of the python function which implements it, in the form
  <module>:<name>. For example, the name
  'datasets.github.scrape_repos.inliners:CxxHeaders' will return the
  function 'CxxHeaders' in the module 'datasets.github.scrape_repos.inliners'.

  Args:
    name: The name of the preprocessor to get.

  Returns:
    The python preprocessor function.

  Raises:
    ValueError: If the requested name cannot be found or is not a
      @dataset_preprocessor decorated function.
  """
  components = name.split(':')
  if len(components) != 2:
    raise ValueError(f'Invalid preprocessor name {name}')
  module_name, function_name = components
  try:
    module = importlib.import_module(module_name)
    function_ = getattr(module, function_name)
  except (ModuleNotFoundError, AttributeError):
    raise ValueError(f'Preprocessor {name} not found.')
  if not function_.__dict__.get('is_dataset_preprocessor'):
    raise ValueError(
        f'Preprocessor {name} not decorated with @dataset_preprocessor')
  return function_


def Preprocess(import_root: pathlib.Path, file_relpath: str,
               all_file_relpaths: typing.List[str],
               preprocessors: typing.List[str]) -> typing.List[str]:
  """Preprocess a text using the given preprocessor pipeline.

  If preprocessing succeeds, the preprocessed text is returned. If preprocessing
  fails (in an expected way, for example by trying to compile incorrect code),
  a BadCodeException is raised. Any other error leads to an InternalError.


  Args:
    import_root: The root of the directory to import the file from.
    file_relpath: The path of the file to import, relative to import_root.
    all_file_relpaths: A list of all paths within the current scope, relative to
      import_root.
    preprocessors: The list of preprocessor functions to run. These will be
      passed to GetPreprocessorFunction() to resolve the python implementations.

  Returns:
    Preprocessed sources.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the requested preprocessors cannot be loaded.
    BadCodeException: If one of the preprocessors rejects the input.
    InternalException: In case of some other error.
  """
  path = import_root / file_relpath
  if not path.is_file():
    raise FileNotFoundError(f"File not found: {path}")

  with open(path) as f:
    texts = [f.read()]

  preprocessor_functions = [GetPreprocessorFunction(p) for p in preprocessors]
  next_texts = []
  for preprocessor in preprocessor_functions:
    for text in texts:
      next_texts += preprocessor(import_root=import_root,
                                 file_relpath=file_relpath,
                                 text=text,
                                 all_file_relpaths=all_file_relpaths)
    texts = next_texts
  return texts
