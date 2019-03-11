"""Tests for //experimental/deeplearning/polyglot/baselines/get_instances.py."""
import tempfile

import pytest

from experimental.deeplearning.polyglot import get_instances
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_GetInstanceConfigs_working_dir():
  """Check the working directory of instance configs."""
  with tempfile.TemporaryDirectory() as working_dir:
    app.FLAGS(['argv[0]', '--working_dir', working_dir])
    assert FLAGS.working_dir == working_dir
    configs = get_instances.GetInstanceConfigs()
    for instance in configs.instance:
      assert instance.working_dir == working_dir


@pytest.mark.skip(reason='Corpus directories do not exist.')
def test_GetInstances():
  """Instantiate all of the instances."""
  with tempfile.TemporaryDirectory() as working_dir:
    app.FLAGS(['argv[0]', '--working_dir', working_dir])
    assert FLAGS.working_dir == working_dir
    assert get_instances.GetInstances()


if __name__ == '__main__':
  test.Main()
