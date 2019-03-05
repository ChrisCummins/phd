"""A CLgen model with backtracking inference."""

import copy
import pathlib
import shutil
import tempfile
import typing

import numpy as np
from absl import flags
from absl import logging

from deeplearning.clgen.preprocessors import preprocessors
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
flags.DEFINE_bool(
    'experimental_clgen_backtracking_lstm_state', False,
    'If set, restore LSTM state tuples during backtracking. '
    'Else re-seed model.')


class OpenClBacktrackingHelper(object):
  """A backtracking helper for OpenCL kernels."""
  END_OF_STATEMENT_TOKEN = ';'

  def __init__(self, atomizer: atomizers.AtomizerBase):
    # Temporary working directory is used to write files that the Grewe feature
    # extractor can use.
    self.working_dir = pathlib.Path(
        tempfile.mkdtemp(prefix='phd_clgen_backtracking_'))
    self.symtok = samplers.SymmetricalTokenDepthCriterion(
        sampler_pb2.SymmetricalTokenDepth(
            depth_increase_token='{', depth_decrease_token='}'))
    self.symtok.Specialize(atomizer)

  def __del__(self):
    shutil.rmtree(self.working_dir)

  def ShouldProceed(self, sample_in_progress: typing.List[str]) -> bool:
    """Determine if a partial sample should be used as the new rollback state.

    Args:
      sample_in_progress: A list of strings, where each string is a token. The
        last token must be ';'.

    Returns:
      True if sampling should proceed with the current partial sample, else
      False.
    """
    candidate_src = self.TryToCloseProgram(sample_in_progress)

    if not candidate_src:
      return False

    # Extract features.
    try:
      path = self.working_dir / 'kernel.cl'
      with open(path, 'w') as f:
        f.write(candidate_src)
      list(grewe_features.ExtractFeaturesFromPath(path))
      # TODO(cec): Record new "best" towards matching a feature vector.
      return True
    except grewe_features.FeatureExtractionError as e:
      return False

  def TryToCloseProgram(
      self, sample_in_progress: typing.List[str]) -> typing.Optional[str]:
    """Try to construct a syntactically valid program from a partial sample.

    Given a partially complete sample, this method attempts to make the smallest
    addition to the code in order to produce a syntactically valid program.

    Args:
      sample_in_progress: A list of strings, where each string is a token. The
        last token must be ';'.

    Returns:
      A string of OpenCL code, if closing the partial sample succeeded, else
      None.
    """
    assert sample_in_progress[-1] == ';'
    bracket_depth = self.symtok.GetTokenDepth(sample_in_progress)
    if bracket_depth > 0:
      return ''.join(sample_in_progress) + ('}' * bracket_depth)


class BacktrackingModel(models.Model):
  """A CLgen model which uses a backtracking approach to sampling."""

  def __init__(self, *args, **kwargs):
    super(BacktrackingModel, self).__init__(*args, **kwargs)
    if not isinstance(self.backend, tensorflow_backend.TensorFlowBackend):
      raise TypeError(f"{self(type).__name__} only compatible with "
                      "TensorFlow backend!")

  def SamplerCache(self, s: samplers.Sampler) -> pathlib.Path:
    """Custom override to prevent cache conflicts with base samplers."""
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

    # We're use the sampler.encoded_start_text attribute as a way to re-seed the
    # model state during rollback, so save the original value here so that we
    # can restore it at the end of the sample batch.
    original_sampler_encoded_start_text = sampler.encoded_start_text.copy()

    self.backend.InitSampleBatch(sampler, batch_size=1)

    backtracker = OpenClBacktrackingHelper(atomizer)
    sampled_tokens = self.SampleOneWithBacktracking(sampler, atomizer,
                                                    backtracker)

    end_time = labdate.MillisecondsTimestamp()

    # Format text.
    text = preprocessors.Preprocess(''.join(sampled_tokens), [
        'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
        'deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype',
        'deeplearning.clgen.preprocessors.common:StripTrailingWhitespace',
        'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
        'deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines',
        'deeplearning.clgen.preprocessors.cxx:ClangFormat',
    ])

    sample = model_pb2.Sample(
        text=text,
        sample_start_epoch_ms_utc=start_time,
        sample_time_ms=end_time - start_time,
        wall_time_ms=end_time - start_time,
        num_tokens=len(sampled_tokens))

    if print_samples:
      print(f'=== CLGEN SAMPLE ===\n\n{text}\n')

    # Restore the sampler's start text.
    sampler.encoded_start_text = original_sampler_encoded_start_text

    return [sample]

  def SampleOneWithBacktracking(
      self, sampler: samplers.Sampler, atomizer: atomizers.AtomizerBase,
      backtracker: OpenClBacktrackingHelper) -> typing.List[str]:
    """Produce a single sample using backtracking.

    Args:
      sampler: The sampler.
      atomizer: The corpus atomizer.
      backtracker: An instance of the backtracking helper class.

    Returns:
      A sample, as a sequence of strings.
    """
    # During sampling, the 'sample_in_progress' contains the current candidate
    # sequence of tokens, and 'rollback_state' contains a copy of
    # 'sample_in_progress' at the last point that
    sample_in_progress = sampler.tokenized_start_text.copy()
    rollback_state = sample_in_progress.copy()
    if FLAGS.experimental_clgen_backtracking_lstm_state:
      rollback_backend_state = copy.deepcopy(self.backend.inference_state)
      rollback_backend_indices = self.backend.inference_indices.copy()

    # This counter is incremented every time ShouldProceed() returns False. If
    # the value grows to exceed the --experimental_clgen_backtracking_attempts
    # flag, backtracking aborts. This is reset when ShouldProceed() returns
    # True.
    backtrack_attempt_count = 0
    # This counter is incremented every time ShouldProceed() returns True. It is
    # used only for logging / debugging.
    checkpoint_count = 0

    while True:
      indices = self.backend.SampleNextIndices(
          sampler, batch_size=1, done=np.array([False]))
      assert len(indices) == 1

      for encoded_token in indices[0]:
        token = atomizer.decoder[encoded_token]
        sample_in_progress.append(token)

        if token == OpenClBacktrackingHelper.END_OF_STATEMENT_TOKEN:
          if backtracker.ShouldProceed(sample_in_progress):
            checkpoint_count += 1
            logging.info(
                'Reached checkpoint %d after %d attempts, '
                '%d tokens', checkpoint_count, backtrack_attempt_count,
                len(sample_in_progress))
            logging.debug("Sample so far << EOF\n%s\nEOF",
                          ''.join(sample_in_progress))
            rollback_state = sample_in_progress.copy()
            # Reset the backtracking ticking clock.
            backtrack_attempt_count = 0
            if FLAGS.experimental_clgen_backtracking_lstm_state:
              rollback_backend_state = copy.deepcopy(
                  self.backend.inference_state)
              rollback_backend_indices = self.backend.inference_indices.copy()
            else:
              # Set the sampler's seed text to be the new rollback state so that
              # when InitSampleBatch() is called during backtracking (below), the
              # model is re-seeded with the entire sample up to this point.
              sampler.encoded_start_text = atomizer.AtomizeString(
                  ''.join(rollback_state))[-sampler.sequence_length:]
          elif (backtrack_attempt_count >=
                FLAGS.experimental_clgen_backtracking_attempts):
            # This branch provides a get-out in case sampling ever gets "stuck".
            # The value of --experimental_clgen_backtracking_attempts should be
            # large enough that this should never realistically happen.
            logging.warning("Crashing out at checkpoint %d after %d attempts",
                            checkpoint_count, backtrack_attempt_count)
            rollback_src = backtracker.TryToCloseProgram(rollback_state)
            assert rollback_src
            return atomizer.TokenizeString(rollback_src)
          else:
            # Backtrack. Re-seed the network with the last
            logging.debug(
                "Backtrack attempt %d rejected at checkpoint %d. "
                "Rejected statement: `%s`", backtrack_attempt_count,
                checkpoint_count, ''.join(
                    sample_in_progress[len(rollback_state):]))
            sample_in_progress = rollback_state.copy()
            if FLAGS.experimental_clgen_backtracking_lstm_state:
              self.backend.inference_state = copy.deepcopy(
                  rollback_backend_state)
              self.backend.inference_indices = rollback_backend_indices.copy()
            else:
              self.backend.InitSampleBatch(sampler, batch_size=1)
            backtrack_attempt_count += 1
            # Tokens are sampled in batches. Don't proceed any further in the
            # batch.
            break
        elif sampler.SampleIsComplete(sample_in_progress):
          logging.info(
              "Reached natural sampling termination at checkpoint %d, "
              "%d tokens", checkpoint_count, len(sample_in_progress))
          return sample_in_progress
