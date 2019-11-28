# DeepSmith python package details
#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
import os
import re
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand


class DsmithTestCommand(TestCommand):
  description = "run test suite"
  user_options = []

  def run_tests(self):
    import experimental.dsmith.test

    experimental.dsmith.test.testsuite()


def read_requirements(path="requirements.txt"):
  if not os.path.exists("requirements.txt"):
    print("please run ./configure first", file=sys.stderr)
    sys.exit(1)

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
  os.chdir(re.sub(r"\.", "/", module))

  # recursively list files in datadir, relative to module root
  files = [
    os.path.join(dp, f)
    for dp, dn, filenames in os.walk(datadir, followlinks=True)
    for f in filenames
    if not any(os.path.join(dp, f).startswith(x) for x in excludes)
  ]

  # restore working directory
  os.chdir(cwd)

  return files


setup(
  name="dsmith",
  version="1.0.0.dev0",
  description="Compiler Fuzzing through Deep Learning",
  url="https://github.com/ChrisCummins/dsmith",
  author="Chris Cummins",
  author_email="chrisc.101@gmail.com",
  license="GNU General Public License, Version 3",
  packages=[
    "dsmith",
    "dsmith.services",
    "dsmith.glsl",
    "dsmith.opencl",
    "dsmith.sol",
    "dsmith.test",
  ],
  package_data={
    "dsmith": all_module_data_files("dsmith"),
    "dsmith.test": all_module_data_files(
      "dsmith.test", excludes=["data/cache",]
    ),
  },
  entry_points={"console_scripts": ["dsmith=dsmith.cli:main"]},
  install_requires=read_requirements("requirements.txt"),
  cmdclass={"test": DsmithTestCommand},
  data_files=[],
  zip_safe=False,
)
