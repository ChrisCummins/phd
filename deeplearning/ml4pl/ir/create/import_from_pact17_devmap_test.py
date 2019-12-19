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
"""Unit tests for //deeplearning/ml4pl/datasets:opencl.py."""
from deeplearning.ml4pl.bytecode.create import (
  import_from_pact17_devmap as opencl,
)
from labm8.py import app
from labm8.py import test


FLAGS = app.FLAGS


@test.Fixture(scope="session")
def opencl_dataset() -> opencl.OpenClDeviceMappingsDataset:
  yield opencl.OpenClDeviceMappingsDataset()


@test.XFail(reason="github.com/ChrisCummins/ProGraML/issues/7")
def test_OpenClDeviceMappingsDataset_cfgs_df_count(
  opencl_dataset: opencl.OpenClDeviceMappingsDataset,
):
  """Test that dataset has expected number of rows."""
  # TODO(github.com/ChrisCummins/ProGraML/issues/7): This doesn't seem to be
  # deterministic.
  assert len(opencl_dataset.cfgs_df) >= 185


def test_OpenClDeviceMappingsDataset_cfgs_df_contains_valid_graphs(
  opencl_dataset: opencl.OpenClDeviceMappingsDataset,
):
  """Test that graph instances are valid."""
  for cfg in opencl_dataset.cfgs_df["cfg:graph"].values:
    assert cfg.ValidateControlFlowGraph(strict=False)


if __name__ == "__main__":
  test.Main()
