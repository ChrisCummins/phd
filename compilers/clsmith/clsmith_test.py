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
"""Unit tests for //compilers/clsmith/clsmith.py."""
import pytest

from compilers.clsmith import clsmith
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_Exec_no_args():
  """Test that CLSmith returns a generated file."""
  src = clsmith.Exec()
  # Check the basic structure of the generated file.
  assert src.startswith("// -g ")
  assert "__kernel void " in src


def test_Exec_invalid_argument():
  """Test that CLSmithError is raised if invalid args passed to CLSmith."""
  with pytest.raises(clsmith.CLSmithError) as e_ctx:
    clsmith.Exec("--invalid_opt")
  assert "" == str(e_ctx.value)
  assert e_ctx.value.returncode == 255


if __name__ == "__main__":
  test.Main()
