# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os

from setuptools import setup


def read_requirements(path='requirements.txt'):
    with open(path) as infile:
        return [x.strip() for x in infile.readlines() if x.strip()]

setup(
    name='gh-archiver',
    version='0.0.2',
    description="Clone and update a GitHub user's repos locally.",
    url='https://github.com/ChrisCummins/gh-archiver',
    author='Chris Cummins',
    author_email='chrisc.101@gmail.com',
    license='MIT License',
    packages=['gh_archiver'],
    entry_points={'console_scripts': ['gh-archiver=gh_archiver:main']},
    install_requires=read_requirements('requirements.txt'),
    zip_safe=True)
