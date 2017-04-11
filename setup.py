# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
from setuptools import setup

with open('./requirements.txt') as infile:
    requirements = [x.strip() for x in infile.readlines() if x.strip()]

setup(name='cldrive',
      version='0.0.9',
      description='Run arbitrary OpenCL kernels',
      url='https://github.com/ChrisCummins/cldrive',
      author='Chris Cummins',
      author_email='chrisc.101@gmail.com',
      license='GNU General Public License, Version 3',
      packages=['cldrive'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest>=3.0'],
      scripts=['bin/cldrive'],
      install_requires=requirements,
      # not zip safe, since we directly invoke the driver module as a script:
      zip_safe=False)
