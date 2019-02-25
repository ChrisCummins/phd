"""Preprocessors to extract from source code."""
import os
import pathlib
import subprocess
import typing

from absl import flags
from absl import logging

from datasets.github.scrape_repos.preprocessors import public
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import bazelutil
from labm8 import pbutil

FLAGS = flags.FLAGS

# The path to the methods extractor binary.
JAVA_METHODS_EXTRACTOR = bazelutil.DataPath(
    'phd/datasets/github/scrape_repos/preprocessors/JavaMethodsExtractor')

# The environments to run the method extractor under.
STATIC_ONLY_ENV = os.environ.copy()
STATIC_ONLY_ENV['JAVA_METHOD_EXTRACTOR_STATIC_ONLY'] = '1'


def ExtractJavaMethods(text: str, static_only: bool = True) -> typing.List[str]:
  """Extract Java methods from a file.

  Args:
    text: The text of the target file.
    static_only: If true, only static methods are returned.

  Returns:
    A list of method implementations.

  Raises:
    ValueError: In case method extraction fails.
  """
  logging.debug('$ %s', JAVA_METHODS_EXTRACTOR)
  process = subprocess.Popen([JAVA_METHODS_EXTRACTOR],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             env=STATIC_ONLY_ENV if static_only else None)
  stdout, stderr = process.communicate(text)
  if process.returncode:
    raise ValueError("JavaMethodsExtractor exited with non-zero "
                     f"status {process.returncode}")
  methods_list = pbutil.FromString(stdout, scrape_repos_pb2.MethodsList())
  return list(methods_list.method)


@public.dataset_preprocessor
def JavaMethods(import_root: pathlib.Path, file_relpath: str, text: str,
                all_file_relpaths: typing.List[str]) -> typing.List[str]:
  """Extract Java methods from a file.

  Args:
    import_root: The root of the directory to import from.
    file_relpath: The path to the target file to import, relative to
      import_root.
    text: The text of the target file.
    all_file_relpaths: A list of all paths within the current scope, relative to
      import_root.

  Returns:
    A list of method implementations.

  Raises:
    ValueError: In case method extraction fails.
  """
  del import_root
  del file_relpath
  del all_file_relpaths
  return ExtractJavaMethods(text, static_only=False)


@public.dataset_preprocessor
def JavaStaticMethods(import_root: pathlib.Path, file_relpath: str, text: str,
                      all_file_relpaths: typing.List[str]) -> typing.List[str]:
  """Extract Java static methods from a file.

  Args:
    import_root: The root of the directory to import from.
    file_relpath: The path to the target file to import, relative to
      import_root.
    text: The text of the target file.
    all_file_relpaths: A list of all paths within the current scope, relative to
      import_root.

  Returns:
    A list of static method implementations.

  Raises:
    ValueError: In case method extraction fails.
  """
  del import_root
  del file_relpath
  del all_file_relpaths
  return ExtractJavaMethods(text, static_only=True)
