"""Utils for handling python modules.
"""
import imp
import sys

import lib.labm8 as lab
from lib.labm8 import io


def import_foreign(name, custom_name=None):
  """
  Import a module with a custom name.

  NOTE this is only needed for Python2. For Python3, import the
  module using the "as" keyword to declare the custom name.

  For implementation details, see:
  http://stackoverflow.com/a/6032023

  Example:

    To import the standard module "math" as "std_math":

        if labm8.is_python3():
          import math as std_math
        else:
          std_math = modules.import_foreign("math", "std_math")

  Arguments:

      name (str): The name of the module to import.
      custom_name (str, optional): The custom name to assign the module to.

  Raises:
      ImportError: If the module is not found.
  """
  if lab.is_python3():
    io.error(("Ignoring attempt to import foreign module '{mod}' "
              "using python version {major}.{minor}"
              .format(mod=name, major=sys.version_info[0],
                      minor=sys.version_info[1])))
    return

  custom_name = custom_name or name

  f, pathname, desc = imp.find_module(name, sys.path[1:])
  module = imp.load_module(custom_name, f, pathname, desc)
  f.close()

  return module
