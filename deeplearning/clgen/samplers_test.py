"""Unit tests for //deeplearning/clgen/sampler.py."""
import pathlib
import sys

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen import samplers
from deeplearning.clgen.models import models
from deeplearning.clgen.proto import internal_pb2
from lib.labm8 import fs
from lib.labm8 import pbutil


@pytest.fixture(scope='function')
def abc_model(abc_model_config):
  return models.Model(abc_model_config)


def test_Sampler_invalid_start_text(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if start_text is not set."""
  del clgen_cache_dir
  abc_sampler_config.ClearField('start_text')
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)
  abc_sampler_config.start_text = ''
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)


def test_Sampler_invalid_batch_size(clgen_cache_dir, abc_sampler_config):
  """Test that an error is thrown if start_text is not set."""
  del clgen_cache_dir
  abc_sampler_config.ClearField('batch_size')
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)
  abc_sampler_config.batch_size = 0
  with pytest.raises(errors.UserError):
    samplers.Sampler(abc_sampler_config)


def test_Sampler_Sample_one_sample(clgen_cache_dir, abc_model,
                                   abc_sampler_config):
  """Test that Sample() produces the expected number of samples."""
  del clgen_cache_dir
  abc_model.Train()
  abc_sampler_config.min_num_samples = 1
  s = samplers.Sampler(abc_sampler_config)
  # Take a single sample.
  s.Sample(abc_model)
  num_contentfiles = len(fs.ls(s.cache(abc_model)["samples"]))
  # Note that the number of contentfiles may be larger than 1, even though we
  # asked for a single sample, since we split the output on the start text.
  assert num_contentfiles >= 1
  s.Sample(abc_model)
  num_contentfiles2 = len(fs.ls(s.cache(abc_model)["samples"]))
  assert num_contentfiles == num_contentfiles2


def test_Sampler_Sample_five_samples(clgen_cache_dir, abc_model,
                                     abc_sampler_config):
  del clgen_cache_dir
  abc_model.Train()
  abc_sampler_config.min_num_samples = 5
  s = samplers.Sampler(abc_sampler_config)
  s.Sample(abc_model)
  num_contentfiles = len(fs.ls(s.cache(abc_model)["samples"]))
  assert num_contentfiles >= 5


def test_Sampler_Sample_return_value(clgen_cache_dir, abc_model,
                                     abc_sampler_config):
  """Test that Sample() returns Sample protos."""
  del clgen_cache_dir
  abc_model.Train()
  abc_sampler_config.min_num_samples = 1
  s = samplers.Sampler(abc_sampler_config)
  samples = s.Sample(abc_model)
  assert len(samples) >= 1
  assert samples[0].text
  assert samples[0].sample_time_ms
  assert samples[0].sample_start_epoch_ms_utc
  proto = pbutil.FromFile(pathlib.Path(fs.ls(s.sample_dir, abspaths=True)[0]),
                          internal_pb2.Sample())
  assert proto == samples[0]


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
