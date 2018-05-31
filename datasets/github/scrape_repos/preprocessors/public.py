"""This file defines the decorator for marking a dataset preprocessor."""
import pathlib
import typing

from absl import flags


FLAGS = flags.FLAGS

# Type hint for a preprocessor function. See @clgen_preprocess for details.
PreprocessorFunction = typing.Callable[[str], str]


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
    'text': str,
    'return': str,
  }
  if typing.get_type_hints(func) != expected_type_hints:
    return_type = expected_type_hints.pop('return').__name__
    expected_args = ', '.join(
        [f'{k}: {v.__name__}' for k, v in expected_type_hints.items()])
    raise TypeError(
        f'Preprocessor {func.__name__} does not have signature '
        f'"def {func.__name__}({expected_args}) -> {return_type}".')
  func.__dict__['is_dataset_preprocessor'] = True
  return func
