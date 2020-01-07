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
"""Unit tests for //compilers/llvm/opt.py."""
from compilers.llvm import opt
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Parametrize("o", ("-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"))
def test_ValidateOptimizationLevel_valid(o: str):
  """Test that valid optimization levels are returned."""
  assert opt.ValidateOptimizationLevel(o) == o


@test.Parametrize(
  "o",
  (
    "O0",  # missing leading '-'
    "-Ofast",  # valid for clang, not for opt
    "-O4",  # not a real value
    "foo",
  ),
)  # not a real value
def test_ValidateOptimizationLevel_invalid(o: str):
  """Test that invalid optimization levels raise an error."""
  with test.Raises(ValueError) as e_ctx:
    opt.ValidateOptimizationLevel(o)
  assert o in str(e_ctx.value)


if __name__ == "__main__":
  test.Main()
