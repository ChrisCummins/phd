# Copyright (c) 2017-2020 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping/models:ncc."""
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import deeptune
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import testlib
from labm8.py import test


def test_model(classify_df, classify_df_atomizer):
  """Run common model tests."""
  testlib.HeterogeneousMappingModelTest(
    deeptune.DeepTune,
    classify_df,
    classify_df_atomizer,
    {
      "lstm_layer_size": 8,
      "dense_layer_size": 4,
      "num_epochs": 2,
      "batch_size": 4,
      "input_shape": (10,),
    },
  )


if __name__ == "__main__":
  test.Main()
