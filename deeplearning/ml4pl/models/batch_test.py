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
"""Unit tests for //deeplearning/ml4pl/models:batch."""
import numpy as np

from deeplearning.ml4pl.models import batch as batches
from labm8.py import test


FLAGS = test.FLAGS


@test.Parametrize("weight", [1, 0.5, 10])
def test_RollingResults_iteration_count(weight: float):
  """Test aggreation of model iteration count and convergence."""
  rolling_results = batches.RollingResults()

  data = batches.Data(graph_ids=[1], data=None)
  results = batches.Results.Create(
    np.random.rand(1, 10),
    np.random.rand(1, 10),
    iteration_count=1,
    model_converged=True,
  )

  for _ in range(10):
    rolling_results.Update(data, results, weight=weight)

  assert rolling_results.iteration_count == 1
  assert rolling_results.model_converged == 1


if __name__ == "__main__":
  test.Main()
