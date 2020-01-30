# Copyright 2019-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //compilers/llvm:llvm_dis."""
from compilers.llvm import llvm_dis
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_Exec_smoke_test():
  """llvm-link with an empty file."""
  p = llvm_dis.Exec(["-help"])
  assert not p.returncode
  assert "USAGE: llvm-dis" in p.stdout


if __name__ == "__main__":
  test.Main()
