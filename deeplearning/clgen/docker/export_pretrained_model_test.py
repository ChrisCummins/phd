"""Unit tests for //deeplearning/clgen/docker:export_pretrained_model."""

import pytest

from deeplearning.clgen import clgen
from deeplearning.clgen.docker import export_pretrained_model
from deeplearning.clgen.proto import clgen_pb2
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def abc_instance(abc_instance_config: clgen_pb2.Instance):
  """Test fixture that yields an instance."""
  return clgen.Instance(abc_instance_config)


def test_TODO(abc_instance: clgen.Instance):
  """TODO: Short summary of test."""
  export_pretrained_model.ExportInstance(abc_instance,
                                         'chriscummins/clgen:latest')
  assert True


if __name__ == '__main__':
  test.Main()
