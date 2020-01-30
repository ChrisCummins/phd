# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test graph builder on IRs that were found to exposure bugs."""
from deeplearning.ml4pl.graphs.llvm2graph import llvm2graph
from labm8.py import bazelutil
from labm8.py import fs
from labm8.py import test

FLAGS = test.FLAGS

MODULE_UNDER_TEST = "deeplearning.ml4pl.graphs.llvm2graph.legacy"

REGRESSION_TESTS = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/testing/data/bytecode_regression_tests"
)


def test_53():
  """github.com/ChrisCummins/ProGraML/issues/53"""
  llvm2graph.BuildProgramGraphNetworkX(fs.Read(REGRESSION_TESTS / "53.ll"))


def test_105975():
  """CFG has BBs without predecessors that need to be removed."""
  llvm2graph.BuildProgramGraphNetworkX(fs.Read(REGRESSION_TESTS / "105975.ll"))


def test_115532():
  """Number of callsites does not correlate with callgraph."""
  llvm2graph.BuildProgramGraphNetworkX(fs.Read(REGRESSION_TESTS / "115532.ll"))


def test_4180():
  """Graph takes more than 120 seconds to construct."""
  llvm2graph.BuildProgramGraphNetworkX(fs.Read(REGRESSION_TESTS / "4180.ll"))


def test_560():
  """Graph has no exit blocks."""
  llvm2graph.BuildProgramGraphNetworkX(fs.Read(REGRESSION_TESTS / "560.ll"))


def test_400531():
  """Graph has no exit blocks."""
  llvm2graph.BuildProgramGraphNetworkX(fs.Read(REGRESSION_TESTS / "400531.ll"))


if __name__ == "__main__":
  test.Main()
