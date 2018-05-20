import os

from pkg_resources import resource_filename, resource_string

from deeplearning.clgen.clgen import File404, InternalError
from lib.labm8 import fs


def must_exist(*path_components: str, **kwargs) -> str:
  """
  Require that a file exists.

  Parameters
  ----------
  *path_components : str
      Components of the path.
  **kwargs
      Key "Error" specifies the exception type to throw.

  Returns
  -------
  str
      Path.
  """
  assert (len(path_components))

  path = os.path.expanduser(os.path.join(*path_components))
  if not os.path.exists(path):
    Error = kwargs.get("Error", File404)
    e = Error("path '{}' does not exist".format(path))
    e.path = path
    raise e
  return path


def package_path(*path) -> str:
  """
  Path to package file.

  Parameters
  ----------
  *path : str
      Path components.

  Returns
  -------
  str
      Path.
  """
  path = os.path.expanduser(os.path.join(*path))
  abspath = resource_filename(__name__, path)
  return must_exist(abspath)


def data_path(*path) -> str:
  """
  Path to package file.

  Parameters
  ----------
  *path : str
      Path components.

  Returns
  -------
  str
      Path.
  """
  return package_path("data", *path)


def package_data(*path) -> bytes:
  """
  Read package data file.

  Parameters
  ----------
  path : str
      The relative path to the data file, e.g. 'share/foo.txt'.

  Returns
  -------
  bytes
      File contents.

  Raises
  ------
  InternalError
      In case of IO error.
  """
  # throw exception if file doesn't exist
  package_path(*path)

  try:
    return resource_string(__name__, fs.path(*path))
  except Exception:
    raise InternalError("failed to read package data '{}'".format(path))


def package_str(*path) -> str:
  """
  Read package data file as a string.

  Parameters
  ----------
  path : str
      The relative path to the text file, e.g. 'share/foo.txt'.

  Returns
  -------
  str
      File contents.

  Raises
  ------
  InternalError
      In case of IO error.
  """
  try:
    return package_data(*path).decode('utf-8')
  except UnicodeDecodeError:
    raise InternalError("failed to decode package data '{}'".format(path))


def sql_script(name: str) -> str:
  """
  Read SQL script to string.

  Parameters
  ----------
  name : str
      The name of the SQL script (without file extension).

  Returns
  -------
  str
      SQL script.
  """
  path = fs.path('data', 'sql', str(name) + ".sql")
  return package_str(path)
