import json
import re

from labm8.py import crypto
from labm8.py import fs


class Error(Exception):
  """
  Module-level base error class.
  """

  pass


class FeatureExtractionError(Error):
  """
  Error thrown if feature extraction fails.
  """

  pass


class ConfigNotFoundError(Error):
  """
  Error thrown if config file is not found.
  """

  pass


def hash_device(*features):
  """
  Returns the hash of a device name + device count pair.
  """
  name = features[0]
  count = features[1]
  return str(count) + "x" + name.strip()


def hash_kernel(*features):
  """
  Returns the hash of a kernel.
  """
  return crypto.sha1(".".join([str(feature) for feature in features]))


def hash_dataset(width, height, tin, tout):
  """
  Returns the hash of a data description.
  """
  return ".".join((str(width), str(height), tin, tout))


def hash_params(wg_c, wg_r):
  """
  Returns the hash of a set of parameter values.
  """
  return str(wg_c) + "x" + str(wg_r)


def unhash_params(params):
  """
  Returns the wg_c and wg_r values of a parameters ID.
  """
  wg_c, wg_r = params.split("x")
  return int(wg_c), int(wg_r)


def hash_scenario(device_id, kernel_id, dataset_id):
  """
  Returns the hash of a scenario.
  """
  return crypto.sha1(".".join((device_id, kernel_id, dataset_id)))


def hash_classifier(classifier):
  """
  Returns the hash of a classifier.
  """
  string = " ".join([classifier.classname,] + classifier.options)
  return re.sub(r"[ -\.]+", "-", string)


def hash_err_fn(err_fn):
  """
  Returns the hash of an error handler partial.
  """
  return err_fn.func.__name__


def hash_ml_dataset(instances):
  """
  Returns the hash of a WekaInstances object.
  """
  return crypto.sha1(str(instances))


def hash_ml_job(name):
  """
  Returns the hash of an ml job name.
  """
  return name


def get_user_source(source):
  """
  Return the user source code for a stencil kernel.

  This strips the common stencil implementation, i.e. the border
  loading logic.

  Raises:
      FeatureExtractionError if the "end of user code" marker is not found.
  """
  lines = source.split("\n")
  user_source = []
  for line in lines:
    if line == "// --- SKELCL END USER CODE ---":
      return "\n".join(user_source)
    user_source.append(line)

  raise FeatureExtractionError("Failed to find end of user code marker")


def get_kernel_name_and_type(source):
  """
  Figure out whether a kernel is synthetic or otherwise.

  Arguments:

      source (str): User source code for kernel.

  Returns:

      (bool, str): Where bool is whether the kernel is synthetic,
        and str is the name of the kernel. If it can't figure out
        the name, returns "unknown".
  """

  def _get_printable_source(lines):
    for i, line in enumerate(lines):
      # Store index of shape define lines.
      if re.search(r"^#define SCL_NORTH", line):
        north = i
      if re.search(r"^#define SCL_SOUTH", line):
        south = i
      if re.search(r"^#define SCL_EAST", line):
        east = i
      if re.search(r"^#define SCL_WEST", line):
        west = i

      # If we've got as far as the user function, then print
      # what we have.
      if re.search("^(\w+) USR_FUNC", line):
        return "\n".join(
          [lines[north], lines[south], lines[east], lines[west], ""] + lines[i:]
        )

    # Fall through, just print the whole bloody lot.
    return "\n".join(lines)

  lines = source.split("\n")

  # Look for clues in the source.
  for line in lines:
    if re.search('^// "Simple" kernel', line):
      return True, "simple"
    if re.search('^// "Complex" kernel', line):
      return True, "complex"

  # Base case, prompt the user.
  print("\nFailed to automatically deduce a kernel name and type.")
  print("Resorting to help from meat space:")
  print("***************** BEGIN SOURCE *****************")
  print(_get_printable_source(lines), "\n")
  name = raw_input("Name me: ")
  synthetic = raw_input("Synthetic? (y/n): ")

  # Sanitise and return user input
  return (
    True if synthetic.strip().lower() == "y" else False,
    name.strip().lower(),
  )


def load_config(path="~/.omnitunerc.json"):
  path = fs.abspath(path)
  if fs.isfile(path):
    return json.load(open(path))
  else:
    raise ConfigNotFoundError("File '{}' not found!".format(path))


def main():
  import server

  server.main()
