"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping/models:ncc."""
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import \
  deeptune
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import testlib
from labm8 import test


def test_model(classify_df, classify_df_atomizer):
  """Run common model tests."""
  testlib.HeterogeneousMappingModelTest(
      deeptune.DeepTune, classify_df, classify_df_atomizer, {
          'lstm_layer_size': 8,
          'dense_layer_size': 4,
          'num_epochs': 2,
          'batch_size': 4,
          'input_shape': (10,),
      })


if __name__ == '__main__':
  test.Main()
