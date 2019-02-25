"""This file defines the decorator for marking a dataset preprocessor."""
import pathlib
import subprocess
import typing

from absl import flags

from labm8 import fs

FLAGS = flags.FLAGS

# Type hint for a preprocessor function. See @clgen_preprocess for details.
PreprocessorFunction = typing.Callable[[str], typing.List[str]]


def dataset_preprocessor(func: PreprocessorFunction) -> PreprocessorFunction:
  """A decorator which marks a function as a dataset preproceessor.

  A preprocessor is accessible using GetPreprocessFunction(), and is a
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
  expected_type_hints = {
      'import_root': pathlib.Path,
      'file_relpath': str,
      'all_file_relpaths': typing.List[str],
      'text': str,
      'return': typing.List[str],
  }
  if typing.get_type_hints(func) != expected_type_hints:
    return_type = expected_type_hints.pop('return').__name__
    expected_args = ', '.join(
        [f'{k}: {v.__name__}' for k, v in expected_type_hints.items()])
    raise TypeError(f'Preprocessor {func.__name__} does not have signature '
                    f'"def {func.__name__}({expected_args}) -> {return_type}".')
  func.__dict__['is_dataset_preprocessor'] = True
  return func


def GetAllFilesRelativePaths(root_dir: pathlib.Path,
                             follow_symlinks: bool = False) -> typing.List[str]:
  """Get relative paths to all files in the root directory.

  Follows symlinks.

  Args:
    root_dir: The directory to find files in.
    follow_symlinks: If true, follow symlinks.

  Returns:
    A list of paths relative to the root directory.

  Raises:
    EmptyCorpusException: If the content files directory is empty.
  """
  with fs.chdir(root_dir):
    cmd = ['find']
    if follow_symlinks:
      cmd.append('-L')
    cmd += ['.', '-type', 'f']
    find_output = subprocess.check_output(cmd).decode('utf-8').strip()
  if find_output:
    # Strip the leading './' from paths.
    return [x[2:] for x in find_output.split('\n')]
  else:
    return []
