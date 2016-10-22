# Copyright (C) 2015, 2016 Chris Cummins.
#
# This file is part of labm8.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Utils for handling python modules.
"""
import imp
import sys

import labm8 as lab
from labm8 import io


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
