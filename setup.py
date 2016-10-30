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
from setuptools import setup


setup(name="labm8",
      version="0.0.6",
      description="Utils for manipulating quantitative experimental data",
      url="https://github.com/ChrisCummins/labm8",
      author="Chris Cummins",
      author_email="chrisc.101@gmail.com",
      license="GNU General Public License, Version 3",
      packages=["labm8"],
      test_suite="nose.collector",
      tests_require=["nose"],
      install_requires=[
          "humanize == 0.5.1",
          "numpy >= 1.10.4",
          "pandas >= 0.19.0",
          "python-dateutil == 2.5.3",
          "pytz == 2016.7",
          "scipy >= 0.16.1",
          "six == 1.10.0",
      ],
      zip_safe=True)
