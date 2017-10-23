# CLgen python package details
#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
import re

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class CLgenTestCommand(TestCommand):
    description = 'run test suite'
    user_options = []

    def run_tests(self):
        import clgen.test
        clgen.test.testsuite()


def read_requirements(path='requirements.txt'):
    with open(path) as infile:
        return [x.strip() for x in infile.readlines() if x.strip()]


def all_module_data_files(module, datadir="data", excludes=[]):
    """
    Find all data files.

    Arguments:
        module (str): name of module
        datadir (str, optional): name of data directory in module
        excludes (str, optional): list of paths to exclude

    Returns:
        str[]: Relative paths to all data files.
    """
    cwd = os.getcwd()

    # change to the module directory, since package_data paths must be relative
    # to this.
    os.chdir(re.sub(r'\.', '/', module))

    # recursively list files in datadir, relative to module root
    files = [
        os.path.join(dp, f) for dp, dn, filenames
        in os.walk(datadir, followlinks=True) for f in filenames
        if not any(os.path.join(dp, f).startswith(x) for x in excludes)]

    # restore working directory
    os.chdir(cwd)

    return files


setup(
    name='CLgen',
    version='0.4.0.dev0',
    description='Deep Learning program generator',
    url='https://github.com/ChrisCummins/clgen',
    author='Chris Cummins',
    author_email='chrisc.101@gmail.com',
    license='GNU General Public License, Version 3',
    packages=[
        'clgen',
        'clgen.test',
    ],
    package_data={
        'clgen': all_module_data_files('clgen', excludes=[
            'data/gpuverify/testsuite',
        ]),
        'clgen.test': all_module_data_files('clgen.test', excludes=[
            'data/cache',
        ]),
    },
    entry_points={'console_scripts': ['clgen=clgen.cli:main']},
    install_requires=read_requirements('requirements.txt'),
    cmdclass={'test': CLgenTestCommand},
    data_files=[],
    zip_safe=False)
