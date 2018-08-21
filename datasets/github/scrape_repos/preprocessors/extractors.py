"""Preprocessors to extract from source code."""
import pathlib
import subprocess
import typing
from absl import flags
from absl import logging
from phd.lib.labm8 import bazelutil
from phd.lib.labm8 import pbutil

from datasets.github.scrape_repos.preprocessors import public
from datasets.github.scrape_repos.proto import scrape_repos_pb2


FLAGS = flags.FLAGS

JAVA_METHODS_EXTRACTOR = bazelutil.DataPath(
    'phd/datasets/github/scrape_repos/preprocessors/JavaMethodsExtractor')


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
  """
  del import_root
  del file_relpath
  del all_file_relpaths

  logging.debug('$ %s', JAVA_METHODS_EXTRACTOR)
  process = subprocess.Popen([JAVA_METHODS_EXTRACTOR], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
  stdout, stderr = process.communicate(text)
  if process.returncode:
    raise ValueError("JavaMethodsExtractor exited with non-zero "
                     f"status {process.returncode}")
  methods_list = pbutil.FromString(stdout, scrape_repos_pb2.MethodsList())
  return list(methods_list.method)
