"""Unit tests for //deeplearning/clgen/sampler.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen import samplers


# AssertConfigIsValid() tests.

def test_AssertConfigIsValid_no_start_text(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if start_text field is not set."""
  del clgen_cache_dir
  # Field not set.
  abc_sampler_config.ClearField('start_text')
  with pytest.raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert 'Sampler.start_text must be a string' == str(e_info.value)
  # Value is an empty string.
  abc_sampler_config.start_text = ''
  with pytest.raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert 'Sampler.start_text must be a string' == str(e_info.value)


def test_AssertConfigIsValid_invalid_batch_size(abc_sampler_config):
  """Test that an error is thrown if batch_size is < 1."""
  # Field not set.
  abc_sampler_config.ClearField('batch_size')
  with pytest.raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.batch_size must be > 0" == str(e_info.value)
  # Value is zero.
  abc_sampler_config.batch_size = 0
  with pytest.raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.batch_size must be > 0" == str(e_info.value)
  # Value is negative.
  abc_sampler_config.batch_size = -1
  with pytest.raises(errors.UserError) as e_info:
    samplers.Sampler(abc_sampler_config)
  assert "Sampler.batch_size must be > 0" == str(e_info.value)


# Sampler.__init__() tests.

def test_Sampler_config_type_error():
  """Test that a TypeError is raised if config is not a Sampler proto."""
  with pytest.raises(TypeError) as e_info:
    samplers.Sampler(1)
  assert "Config must be a Sampler proto. Received: 'int'" == str(e_info.value)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
