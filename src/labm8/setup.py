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
from setuptools import setup
from sys import version_info

def program_exists(program):
    """
    Return if a program exists in $PATH.

    Arguments:

        program (str): Name of program.

    Returns:

        bool: True if program name found in system path, else false.
    """
    import os
    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = os.path.join(path, program)
        if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
            return True
    return False

# There are two separate packages implementing a wrapper around
# weka. Both require weka to be installed.
if program_exists("weka"):
    if version_info[0] == 2:
        python_weka_wrapper = "python-weka-wrapper"
    else:
        python_weka_wrapper = "python-weka-wrapper3"

setup(name="labm8",
      version="0.0.1",
      description=("A collection of utilities for collecting and "
                   "manipulating quantitative experimental data"),
      url="https://github.com/ChrisCummins/labm8",
      author="Chris Cummins",
      author_email="chrisc.101@gmail.com",
      license="GPL v3",
      packages=["labm8"],
      test_suite="nose.collector",
      tests_require=[
          # FIXME: Add coverage support for parent repo: "coverage",
          "nose"
      ],
      install_requires=[
          "numpy",
          "pandas",
          "scipy",
          python_weka_wrapper
      ],
      zip_safe=False)
