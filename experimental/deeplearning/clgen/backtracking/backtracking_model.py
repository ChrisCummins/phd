"""A CLgen model with backtracking inference."""

import pathlib
import shutil
import tempfile
import typing

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

flags.DEFINE_integer('experimental_clgen_backtracking_attempts', 250,
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
    assert sample_in_progress[-1] == ';'
    bracket_depth = self.symtok.GetTokenDepth(sample_in_progress)
    if bracket_depth > 0:
      return ''.join(sample_in_progress) + ('}' * bracket_depth)

  def ShouldProceed(self, sample_in_progress: typing.List[str]) -> bool:
    candidate_src = self.TryToCloseProgram(sample_in_progress)

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
    """Run a single iteration of the batched sample inner-loop."""
    start_time = labdate.MillisecondsTimestamp()

    if batch_size != 1:  # No support for batched inference.
      raise TypeError(f"Batch size {batch_size} != 1")

    # We're going to be modifying the sampler.encoded_start_text attribute,
    # so save the original value here.
    original_sampler_encoded_start_text = sampler.encoded_start_text.copy()

    sample_in_progress = sampler.tokenized_start_text.copy()
    rollback_state = sample_in_progress.copy()

    backtracker = Backtracker(atomizer)
    backtrack_attempt_count = 0

    self.backend.InitSampleBatch(sampler, batch_size=1)
    done = False

    i = 0
    while not done:
      i += 1
      indices = self.backend.SampleNextIndices(
          sampler, batch_size=1, done=np.array([False]))
      assert len(indices) == 1

      for j, index in enumerate(indices[0]):
        token = atomizer.decoder[index]
        sample_in_progress.append(token)

        if token == Backtracker.END_OF_STATEMENT_TOKEN:
          if backtracker.ShouldProceed(sample_in_progress):
            logging.info(
                'i=%d j=%d Reached new backtrack state after %d attempts, %d tokens',
                i, j, backtrack_attempt_count, len(rollback_state))
            logging.debug("Sample so far: `%s`", ''.join(sample_in_progress))
            rollback_state = sample_in_progress.copy()
            sampler.encoded_start_text = atomizer.AtomizeString(
                ''.join(rollback_state))
          elif backtrack_attempt_count >= FLAGS.experimental_clgen_backtracking_attempts:
            logging.warning(
                "i=%d j=%d Crashing out of backtracking after %d attempts", i,
                j, backtrack_attempt_count)
            done = True
            break
          else:
            # Backtrack.
            self.backend.InitSampleBatch(sampler, batch_size=1)
            backtrack_attempt_count += 1
            logging.debug(
                "i=%d j=%d Backtrack attempt %d! Rejected candidate statement: `%s`",
                i, j, backtrack_attempt_count, ''.join(
                    sample_in_progress[len(rollback_state):]))
            sample_in_progress = rollback_state.copy()
            break
        elif sampler.SampleIsComplete(sample_in_progress):
          logging.info("i=%d j=%d Reached natural sampling termination", i, j)
          done = True
          break

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
    sampler.encoded_start_text = original_sampler_encoded_start_text

    return [sample]
