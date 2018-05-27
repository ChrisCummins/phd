"""Unit tests for //deeplearning/clgen/sampler.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen import samplers


def test_Sampler_config_type_error():
  """Test that a TypeError is raised if config is not a Sampler proto."""
  with pytest.raises(TypeError) as e_info:
    samplers.Sampler(1)
  assert "Config must be a Sampler proto. Received: 'int'" == str(e_info.value)


def test_Sampler_no_start_text_field(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if start_text field is not set."""
  del clgen_cache_dir
  abc_sampler_config.ClearField('start_text')
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)


def test_Sampler_empty_start_text(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if start_text is empty."""
  del clgen_cache_dir
  abc_sampler_config.start_text = ''
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)


def test_Sampler_no_batch_size_field(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if batch_size field is not set."""
  del clgen_cache_dir
  abc_sampler_config.ClearField('batch_size')
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)


def test_Sampler_invalid_batch_size_field(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if batch_size is < 1."""
  del clgen_cache_dir
  abc_sampler_config.batch_size = 0
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)
  abc_sampler_config.batch_size = -1
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
