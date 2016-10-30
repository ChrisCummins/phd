# CLgen python package details
#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
import os
import pip

from setuptools import setup
from setuptools.command.install import install
from pip.req import parse_requirements

# Because of a bug in scipy setup.py, we have to use 'pip install' to install
# dependencies rather than using setuptools. See:
#     https://github.com/scikit-learn/scikit-learn/issues/4164
#
install_reqs = parse_requirements('./requirements.txt', session=False)
requirements = [str(ir.req) for ir in install_reqs]


def all_module_data_files(module, datadir="data"):
    """
    Find all data files.

    Returns:
        str[]: Relative paths to all data files.
    """
    cwd = os.getcwd()

    # change to the module directory, since package_data paths must be relative
    # to this.
    os.chdir(module)

    # recursively list files in datadir, relative to module root
    files = [
        os.path.join(dp, f) for dp, dn, filenames
        in os.walk(datadir, followlinks=True) for f in filenames]

    # restore working directory
    os.chdir(cwd)

    return files

setup(
    name='CLgen',
    version='0.0.1',
    description='',
    url='https://github.com/ChrisCummins/clgen',
    author='Chris Cummins',
    author_email='chrisc.101@gmail.com',
    license='GNU General Public License, Version 3',
    packages=['clgen'],
    package_data={'clgen': all_module_data_files('clgen')},
    scripts=[
        'bin/clgen',
        'bin/clgen-create-db',
        'bin/clgen-drive',
        'bin/clgen-explore',
        'bin/clgen-features',
        'bin/clgen-fetch',
        'bin/clgen-fetch-clgen',
        'bin/clgen-fetch-clsmith',
        'bin/clgen-fetch-db',
        'bin/clgen-fetch-github',
        'bin/clgen-preprocess',
        'bin/clgen-train',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=requirements,
    data_files=[],
    zip_safe=False)
