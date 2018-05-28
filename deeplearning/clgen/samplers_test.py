"""Unit tests for //deeplearning/clgen/sampler.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen import samplers
from deeplearning.clgen.proto import sampler_pb2


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


# MaxlenTerminationCriterion tests.

def test_MaxlenTerminationCriterion_invalid_maximum_tokens_in_sample():
  """Test that error is raised if maximum_tokens_in_sample is invalid."""
  config = sampler_pb2.MaxTokenLength()
  # Field is missing.
  with pytest.raises(errors.UserError) as e_info:
    samplers.MaxlenTerminationCriterion(config)
  assert "MaxTokenLength.maximum_tokens_in_sample must be > 0" == str(
      e_info.value)
  # Value is zero.
  config.maximum_tokens_in_sample = 0
  with pytest.raises(errors.UserError) as e_info:
    samplers.MaxlenTerminationCriterion(config)
  assert "MaxTokenLength.maximum_tokens_in_sample must be > 0" == str(
      e_info.value)


def test_MaxlenTerminationCriterion_SampleIsComplete():
  """Test SampleIsComplete() returns expected values."""
  t = samplers.MaxlenTerminationCriterion(sampler_pb2.MaxTokenLength(
      maximum_tokens_in_sample=3))
  assert not t.SampleIsComplete([])
  assert not t.SampleIsComplete(['a'])
  assert not t.SampleIsComplete(['a', 'b'])
  assert t.SampleIsComplete(['a', 'b', 'c'])
  assert t.SampleIsComplete(['a', 'b', 'c', 'd'])
  assert t.SampleIsComplete(['a', 'b', 'c', 'd', 'e'])


# SymmetricalTokenDepthCriterion tests.

def test_SymmetricalTokenDepthCriterion_depth_increase_token():
  """Test that error is raised if depth_increase_token is invalid."""
  config = sampler_pb2.SymmetricalTokenDepth(depth_decrease_token='a')
  # Field is missing.
  with pytest.raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert 'SymmetricalTokenDepth.depth_increase_token must be a string' == str(
      e_info.value)
  # Value is empty.
  config.depth_increase_token = ''
  with pytest.raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert 'SymmetricalTokenDepth.depth_increase_token must be a string' == str(
      e_info.value)


def test_SymmetricalTokenDepthCriterion_depth_increase_token():
  """Test that error is raised if depth_increase_token is invalid."""
  config = sampler_pb2.SymmetricalTokenDepth(depth_increase_token='a')
  # Field is missing.
  with pytest.raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert 'SymmetricalTokenDepth.depth_decrease_token must be a string' == str(
      e_info.value)
  # Value is empty.
  config.depth_decrease_token = ''
  with pytest.raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert 'SymmetricalTokenDepth.depth_decrease_token must be a string' == str(
      e_info.value)


def test_SymmetricalTokenDepthCriterion_same_tokens():
  """test that error is raised if depth tokens are the same."""
  config = sampler_pb2.SymmetricalTokenDepth(
      depth_increase_token='a', depth_decrease_token='a')
  with pytest.raises(errors.UserError) as e_info:
    samplers.SymmetricalTokenDepthCriterion(config)
  assert 'SymmetricalTokenDepth tokens must be different' == str(e_info.value)


def test_SymmetricalTokenDepthCriterion_SampleIsComplete():
  """Test SampleIsComplete() returns expected values."""
  t = samplers.SymmetricalTokenDepthCriterion(sampler_pb2.SymmetricalTokenDepth(
      depth_increase_token='+', depth_decrease_token='-'))
  assert not t.SampleIsComplete([])
  assert not t.SampleIsComplete(['+'])
  assert not t.SampleIsComplete(['-'])
  assert t.SampleIsComplete(['+', '-'])
  assert not t.SampleIsComplete(['a', '+', 'b', 'c'])
  assert not t.SampleIsComplete(['a', '+', '+', 'b', 'c', '-'])
  assert t.SampleIsComplete(['a', '+', '-', '+', 'b', 'c', '-'])


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
