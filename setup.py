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
import pip
from setuptools import setup
from setuptools.command.install import install
from pip.req import parse_requirements

# Because of a bug in scipy setup.py, we have to use 'pip install' to install
# dependencies rather than using setuptools. See:
#     https://github.com/scikit-learn/scikit-learn/issues/4164
#
install_reqs = parse_requirements('./requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(name="labm8",
      version="0.0.2",
      description="Utils for manipulating quantitative experimental data",
      url="https://github.com/ChrisCummins/labm8",
      author="Chris Cummins",
      author_email="chrisc.101@gmail.com",
      license="GNU General Public License, Version 3",
      packages=["labm8"],
      test_suite="nose.collector",
      tests_require=["nose"],
      install_requires=reqs,
      zip_safe=True)
