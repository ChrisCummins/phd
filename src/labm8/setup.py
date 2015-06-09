# Copyright (C) 2015 Chris Cummins.
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

# Python weka wrapper currently only supports Python2.
python_weka_wrapper = "python-weka-wrapper" if version_info[0] == 2 else ""

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
          "coverage",
          "nose"
      ],
      install_requires=[
          "numpy",
          "pandas",
          "scipy",
          python_weka_wrapper
      ],
      zip_safe=False)
