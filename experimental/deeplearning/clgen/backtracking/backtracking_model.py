"""A CLgen model with backtracking inference."""

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

FLAGS = flags.FLAGS

flags.DEFINE_integer('experimental_clgen_backtracking_attempts', 100,
                     'The number of attempts to make when backtracking.')


class Backtracker(object):
  END_OF_STATEMENT_TOKEN = ';'

  @staticmethod
  def TryToCloseProgram(
      sample_in_progress: typing.List[str],
      symtok: samplers.SymmetricalTokenDepthCriterion) -> typing.Optional[str]:
    bracket_depth = symtok.GetTokenDepth(sample_in_progress)
    if bracket_depth > 0:
      return ''.join(sample_in_progress) + ('}' * bracket_depth)

  @classmethod
  def ShouldProceed(cls, self, sample_in_progress: typing.List[str],
                    symtok: samplers.SymmetricalTokenDepthCriterion) -> bool:
    candidate_src = cls.TryToCloseProgram(sample_in_progress, symtok)

    if not candidate_src:
      return False

    if not cls.Compiles(candidate_src):
      return False

    # TODO(cec): Check if progress is made towards goal.
    return True


class BacktrackingModel(models.Model):

  def __init__(self, *args, **kwargs):
    super(BacktrackingModel, self).__init__(*args, **kwargs)

    if not isinstance(self.backend, tensorflow_backend.TensorFlowBackend):
      raise TypeError(f"{self(type).__name__} only compatible with "
                      "TensorFlow backend!")

  @staticmethod
  def _ComputeHash(*args, **kwargs) -> str:
    """Override to prevent name conflicts with default model."""
    original_hash = BacktrackingModel._ComputeHash(*args, **Kwargs)
    return f'backtracking_{original_hash}'

  def _SampleBatch(self,
                   sampler: samplers.Sampler,
                   atomizer: atomizers.AtomizerBase,
                   batch_size: int,
                   print_samples: typing.Optional[bool] = False
                  ) -> typing.List[model_pb2.Sample]:
    """Implementation of backtracking sampling."""
    start_time = labdate.MillisecondsTimestamp()

    assert batch_size == 1  # No support for batched inference.
    sample_in_progress = sampler.tokenized_start_text.copy()
    backtrack_state = sampler.encoded_start_text.copy()
    original_backtrack_state = sampler.encoded_start_text.copy()

    symtok = samplers.SymmetricalTokenDepthCriterion(
        sampler_pb2.SymmetricalTokenDepth(
            depth_increase_token='{', depth_decrease_token='}'))
    symtok.Specialize(atomizer)

    self.backend.InitSampleBatch(sampler, batch_size=1)
    backtrack_attempt_count = 0

    # Sampling loop. Continues until all samples in the batch are done.
    while not sampler.SampleIsComplete(sample_in_progress):
      indices = self.backend.SampleNextIndices(
          sampler, batch_size=1, done=np.array([0]))
      assert len(indices) == 1  # No support for batched inference.
      assert len(indices[0] == 1)  # No support for multi-indices inference.

      index = indices[0]
      token = atomizer.decoder[index]
      sample_in_progress.append(token)
      if token == Backtracker.END_OF_STATEMENT_TOKEN:
        if Backtracker.ShouldProceed(sample_in_progress):
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
