from __future__ import print_function

import sys
from setuptools import setup

if sys.version_info < (3, 0):
    print("fatal: shutterbug requires Python 3")
    sys.exit(1)

setup(
    name='shutterbug',
    version='0.0.3',
    description='',
    url='https://github.com/ChrisCummins/shutterbug',
    author='Chris Cummins',
    author_email='chrisc.101@gmail.com',
    license='GNU General Public License, Version 3',
    packages=['shutterbug'],
    scripts=['bin/shutterbug'],
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=[],
    data_files=[],
    zip_safe=True)
