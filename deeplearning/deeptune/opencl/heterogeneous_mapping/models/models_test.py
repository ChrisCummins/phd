"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:models."""

from deeplearning.deeptune.opencl.heterogeneous_mapping.models import models
from labm8 import test


def test_num_models():
  """Test that the number of models. This will change"""
  assert len(models.ALL_MODELS) == 5


if __name__ == '__main__':
  test.Main()
