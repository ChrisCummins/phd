# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //compilers/llvm/util.py."""
import typing

import pytest

from compilers.llvm import llvm
from compilers.llvm import util
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


@pytest.mark.parametrize('cflags', [['-O0'], ['-O1'], ['-O2'], ['-O3']])
def test_GetOptArgs_black_box(cflags: typing.List[str]):
  """Black box opt args test."""
  args = util.GetOptArgs(cflags)
  assert args
  for invocation in args:
    assert invocation


def test_GetOptArgs_bad_args():
  """Error is raised if invalid args are passed."""
  with pytest.raises(llvm.LlvmError):
    util.GetOptArgs(['-not-a-real-arg!'])


if __name__ == '__main__':
  test.Main()
