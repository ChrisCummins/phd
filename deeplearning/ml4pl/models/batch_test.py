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
