# Copyright (c) 2017, 2018, 2019 Chris Cummins.
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
"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping/models:grewe."""
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import \
  grewe
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import testlib
from labm8 import test


def test_model(classify_df, classify_df_atomizer):
  """Run common model tests."""
  testlib.HeterogeneousMappingModelTest(grewe.Grewe, classify_df,
                                        classify_df_atomizer, {})


if __name__ == '__main__':
  test.Main()
