"""A CLgen model with backtracking inference."""

import pathlib
import typing
import tempfile
import shutil

import numpy as np
from absl import flags
from absl import logging

from deeplearning.clgen import samplers
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.models import models
from deeplearning.clgen.models import tensorflow_backend
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8 import labdate
from research.grewe_2013_cgo import feature_extractor as grewe_features

FLAGS = flags.FLAGS

flags.DEFINE_integer('experimental_clgen_backtracking_attempts', 100,
                     'The number of attempts to make when backtracking.')


class Backtracker(object):
  END_OF_STATEMENT_TOKEN = ';'

  def __init__(self, atomizer: atomizers.AtomizerBase):
    self.working_dir = pathlib.Path(
        tempfile.mkdtemp(prefix='phd_clgen_backtracking_'))
    self.symtok = samplers.SymmetricalTokenDepthCriterion(
        sampler_pb2.SymmetricalTokenDepth(
            depth_increase_token='{', depth_decrease_token='}'))
    self.symtok.Specialize(atomizer)

  def __del__(self):
    shutil.rmtree(self.working_dir)

  def TryToCloseProgram(
      self, sample_in_progress: typing.List[str]) -> typing.Optional[str]:
    bracket_depth = self.symtok.GetTokenDepth(sample_in_progress)
    if bracket_depth > 0:
      return ''.join(sample_in_progress) + ('}' * bracket_depth)

  def ShouldProceed(self, sample_in_progress: typing.List[str]) -> bool:
    candidate_src = self.TryToCloseProgram(sample_in_progress, self.symtok)

    if not candidate_src:
      return False

    # Extract features.
    try:
      path = self.working_dir / 'kernel.cl'
      with open(path, 'w') as f:
        f.write(candidate_src)
      list(grewe_features.ExtractFeaturesFromPath(path))
      return True
    except grewe_features.FeatureExtractionError as e:
      return False


class BacktrackingModel(models.Model):

  def __init__(self, *args, **kwargs):
    super(BacktrackingModel, self).__init__(*args, **kwargs)

    if not isinstance(self.backend, tensorflow_backend.TensorFlowBackend):
      raise TypeError(f"{self(type).__name__} only compatible with "
                      "TensorFlow backend!")

  def SamplerCache(self, s: samplers.Sampler) -> pathlib.Path:
    """Custom override to prevent cache conflicts with base samplers.

    Args:
      s: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist - it is
      created only after a call to Sample().
    """
    return self.cache.path / 'samples' / f'backtracking_{s.hash}'

  def _SampleBatch(self,
                   sampler: samplers.Sampler,
                   atomizer: atomizers.AtomizerBase,
                   batch_size: int,
                   print_samples: typing.Optional[bool] = False
                  ) -> typing.List[model_pb2.Sample]:
    """Implementation of backtracking sampling."""
    start_time = labdate.MillisecondsTimestamp()

    if batch_size != 1:  # No support for batched inference.
      raise TypeError(f"Batch size {batch_size} != 1")
    sample_in_progress = sampler.tokenized_start_text.copy()
    backtrack_state = sampler.encoded_start_text.copy()
    original_backtrack_state = sampler.encoded_start_text.copy()

    backtracker = Backtracker(atomizer)

    self.backend.InitSampleBatch(sampler, batch_size=1)
    backtrack_attempt_count = 0

    while True:  #not sampler.SampleIsComplete(sample_in_progress):
      logging.info("INFER!")
      indices = self.backend.SampleNextIndices(
          sampler, batch_size=1, done=np.array([0]))
      assert len(indices) == 1  # No support for batched inference.
      assert len(indices[0] == 1)  # No support for multi-indices inference.

      index = indices[0]
      token = atomizer.decoder[index]
      sample_in_progress.append(token)
      if token == Backtracker.END_OF_STATEMENT_TOKEN:
        if backtracker.ShouldProceed(sample_in_progress):
          backtrack_state = sample_in_progress
          logging.debug(
              'Reached new backtrack state after %d attempts, %d tokens',
              backtrack_attempt_count, len(backtrack_state))
          sampler.encoded_start_text = atomizer.AtomizeString(
              ''.join(backtrack_state))
        elif backtrack_attempt_count >= FLAGS.experimental_clgen_backtracking_attempts:
          logging.warning("Crashing out of backtracking after %d attempts",
                          backtrack_attempt_count)
          break
        else:
          # Backtrack.
          self.backend.InitSampleBatch(sampler, batch_size=1)
          backtrack_attempt_count += 1
          logging.info("Backtrack attempt %d!", backtrack_attempt_count)

    end_time = labdate.MillisecondsTimestamp()
    sample = model_pb2.Sample(
        text=''.join(sample_in_progress),
        sample_start_epoch_ms_utc=start_time,
        sample_time_ms=end_time - start_time,
        wall_time_ms=end_time - start_time,
        num_tokens=len(sample_in_progress))

    if print_samples:
      print(f'=== CLGEN SAMPLE ===\n\n{sample.text}\n')

    # Restore the sampler's start text.
    sampler.encoded_start_text = original_backtrack_state

    return [sample]
