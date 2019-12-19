# Copyright 2019 the ProGraML authors.
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
"""Unit tests for //deeplearning/ml4pl/datasets:linux.py."""
from deeplearning.ml4pl.bytecode.create import import_from_linux as linux
from labm8.py import app
from labm8.py import test


FLAGS = app.FLAGS


@test.Fixture(scope="session")
def linux_dataset() -> linux.LinuxSourcesDataset():
  yield linux.LinuxSourcesDataset()


def test_BytecodeFromLinuxSrc_known_file():
  """Test that a known file produces bytecode."""
  path = linux.LinuxSourcesDataset().src_tree_root / "kernel" / "exit.c"
  assert path.is_file()  # If this fails, the linux source tree is broken.

  bytecode = linux.BytecodeFromLinuxSrc(path, "-O0")
  assert bytecode


def test_TryToCreateControlFlowGraphsFromLinuxSrc_known_file():
  """Test that a known file produces graphs."""
  path = linux.LinuxSourcesDataset().src_tree_root / "kernel" / "exit.c"
  assert path.is_file()  # If this fails, the linux source tree is broken.

  # TODO(github.com/ChrisCummins/ProGraML/issues/7): No stable value.
  assert len(linux.TryToCreateControlFlowGraphsFromLinuxSrc(path)) < 20


def test_TryToCreateControlFlowGraphsFromLinuxSrc_graphs_are_valid():
  """Test that a known file produces valid graphs."""
  path = linux.LinuxSourcesDataset().src_tree_root / "kernel" / "exit.c"
  assert path.is_file()  # If this fails, the linux source tree is broken.

  for graph in linux.TryToCreateControlFlowGraphsFromLinuxSrc(path):
    assert graph.IsValidControlFlowGraph(strict=False)


def test_LinuxSourcesDataset_df_count(linux_dataset: linux.LinuxSourcesDataset):
  """Test that dataset has expected number of rows."""
  # TODO(github.com/ChrisCummins/ProGraML/issues/7): This doesn't seem to be
  # deterministic.
  assert len(linux_dataset.cfgs_df) >= 1600


if __name__ == "__main__":
  test.Main()
