"""Unit tests for //deeplearning/clgen/docker:export_pretrained_model."""
import pathlib

import pytest

from deeplearning.clgen import clgen
from deeplearning.clgen.docker import export_pretrained_model
from deeplearning.clgen.proto import clgen_pb2
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def abc_instance(abc_instance_config: clgen_pb2.Instance):
  """Test fixture that yields an instance."""
  return clgen.Instance(abc_instance_config)


def test_ExportInstance(abc_instance: clgen.Instance, tempdir: pathlib.Path):
  export_pretrained_model.ExportInstance(abc_instance, tempdir,
                                         'chriscummins/clgen:latest')
  assert (tempdir / 'Dockerfile').is_file()
  assert (tempdir / 'config.pbtxt').is_file()
  assert (tempdir / 'model').is_dir()


if __name__ == '__main__':
  test.Main()
