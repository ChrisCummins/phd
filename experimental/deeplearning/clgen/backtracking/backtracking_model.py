"""A CLgen model with backtracking inference."""

import copy
import pathlib
import random
import re
import shutil
import tempfile
import typing

import numpy as np
import scipy

from deeplearning.clgen import errors as clgen_errors
from deeplearning.clgen import samplers
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.models import models
from deeplearning.clgen.models import tensorflow_backend
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8 import app
from labm8 import labdate
from research.grewe_2013_cgo import feature_extractor as grewe_features

FLAGS = app.FLAGS

app.DEFINE_integer('experimental_clgen_backtracking_max_attempts', 1000,
                   'The maximum number of attempts to make when backtracking.')
app.DEFINE_integer(
    'experimental_clgen_backtracking_max_steps', 10000,
    'The maximum number of checkpoints to make when backtracking.')
app.DEFINE_float(
    'experimental_clgen_backtracking_reject_no_progress_probability', 0.5,
    'The probability that a step which does not improve feature distance is '
    'rejected. A higher value means that a larger fraction of steps must '
    'directly contribute towards an improved feature distance.')
app.DEFINE_boolean(
    'experimental_clgen_backtracking_lstm_state', False,
    'If set, restore LSTM state tuples during backtracking. '
    'Else re-seed model.')
app.DEFINE_string(
    'experimental_clgen_backtracking_target_features', None,
    'A comma-separated list of four target feature values. If not set, no '
    'target features are used.')
app.DEFINE_float(
    'experimental_clgen_backtracking_max_feature_distance', 0.1,
    'Maximum difference between current and target features before sampling '
    'may terminate.')
app.DEFINE_float(
    'experimental_clgen_backtracking_max_norm_feature_distance', 0.01,
    'Maximum difference between current and target features before sampling '
    'may terminate. The value is normalized to the starting difference, were '
    '1.0 is the starting difference and 0.0 is an exact match.')


class OpenClBacktrackingHelper(object):
  """A backtracking helper for OpenCL kernels."""
  # We want to checkpoint at the end of every logicial statement. An easy way
  # to get us most of the way there is to checkpoint when the last produced
  # character is ';', however, "for" loop syntax provides two exceptions. Given
  # the example:
  #   for (int i = 0; i < 10; ++i) { int x = 10; }
  # there is only a single logical statement, "int x = 10;". A crude workaround
  # to prevent logical statement ends being triggered within the for loop
  # header is to use a pair of regexes to detect them:
  END_OF_STATEMENT_TOKEN = ';'
  FOR_LOOP_1 = re.compile(r'(.|\n)*for(\s|\n)*\([^;]*;')
  FOR_LOOP_2 = re.compile(r'(.|\n)*for(\s|\n)*\([^;]*;[^;]*;')

  def __init__(self, atomizer: atomizers.AtomizerBase,
               target_features: typing.Optional[np.array]):
    # Temporary working directory is used to write files that the Grewe feature
    # extractor can use.
    self.working_dir = pathlib.Path(
        tempfile.mkdtemp(prefix='phd_clgen_backtracking_'))
    self.symtok = samplers.SymmetricalTokenDepthCriterion(
        sampler_pb2.SymmetricalTokenDepth(
            depth_increase_token='{', depth_decrease_token='}'))
    self.symtok.Specialize(atomizer)

    # Feature hill climbing state.
    self._target_features = target_features
    self._previous_features = np.array([0, 0, 0, 0], dtype=np.int)
    self._init_feature_distance = scipy.spatial.distance.euclidean(
        self._previous_features, self._target_features)
    self._previous_src = None
    self._previous_feature_distance = self._init_feature_distance

  def __del__(self):
    shutil.rmtree(self.working_dir)

  def ShouldCheckpoint(self, sampled_token: str) -> bool:
    """Determine whether ShouldProceed() should be called.

    Args:
      sampled_token: The newly sampled token.

    Returns:
      True if ShouldProceed() should be called, else False.
    """
    return sampled_token[-1] == self.END_OF_STATEMENT_TOKEN

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
      # Was unable to create a syntactically valid program from the partial
      # sample.
      return False

    # Feature extractor reads from files.
    path = self.working_dir / 'kernel.cl'
    with open(path, 'w') as f:
      f.write(candidate_src)

    features = self.TryToExtractFeatures(path)
    if features is None:
      # Was unable to extract features from the partial sample.
      return False

    # Grewe feature extractor is robust to code that doesn't compile (i.e. code
    # containing implicit declarations). Run the code through clang to check
    # if it actually compiles, else reject it.
    try:
      opencl.Compile(candidate_src)
    except clgen_errors.ClangException:
      return False

    # Implement pure hill climbing approach to match a target feature vector.
    # When enabled, partial samples which increase the distance to the target
    # feature vector are rejected.
    if self._target_features is not None:
      new_feature_distance = scipy.spatial.distance.euclidean(
          features, self._target_features)
      app.Log(1, 'Features: %s, distance=%f, norm=%f, delta=%f', features,
               new_feature_distance,
               new_feature_distance / self._init_feature_distance,
               new_feature_distance - self._previous_feature_distance)
      if new_feature_distance > self._previous_feature_distance:
        # This will only happen once feature values are great than target
        # feature values.
        app.Log(1, "Rejecting candidate because of positive feature delta")
        return False
      if (new_feature_distance == self._previous_feature_distance and
          random.random() >
          FLAGS.experimental_clgen_backtracking_reject_no_progress_probability):
        app.Log(1, "Randomly rejecting candidate with no progress")
        return False
      self._previous_features = features
      self._previous_src = candidate_src
      self._previous_feature_distance = new_feature_distance

    return True

  def TryToExtractFeatures(self, path: pathlib.Path) -> typing.Optional[str]:
    """ """
    try:
      features = list(grewe_features.ExtractFeaturesFromPath(path))
      if len(features) != 1:
        # It is possible to bleed from one kernel to the next. Treat that as an
        # error.
        return None
      return np.array([
          features[0].compute_operation_count,
          features[0].global_memory_access_count,
          features[0].local_memory_access_count,
          features[0].coalesced_memory_access_count,
      ],
                      dtype=int)
    except grewe_features.FeatureExtractionError as e:
      pass

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
    if sample_in_progress[-1] != self.END_OF_STATEMENT_TOKEN:
      return None
    bracket_depth = self.symtok.GetTokenDepth(sample_in_progress)
    if bracket_depth > 0:
      text = ''.join(sample_in_progress)
      if re.fullmatch(self.FOR_LOOP_1, text):
        return text + ';){}' + '}' * bracket_depth
      elif re.fullmatch(self.FOR_LOOP_2, text):
        return text + '){}' + '}' * bracket_depth
      else:
        return text + '}' * bracket_depth

  def IsDone(self):
    """Return whether sampling is done."""
    if self._target_features is None:
      return True
    else:
      return ((self._previous_feature_distance <=
               FLAGS.experimental_clgen_backtracking_max_feature_distance) or
              (self._previous_feature_distance / self._init_feature_distance <=
               FLAGS.experimental_clgen_backtracking_max_norm_feature_distance))

  @property
  def target_features(self) -> np.array:
    return self._target_features

  @property
  def current_features(self) -> np.array:
    return self._previous_features

  @property
  def feature_distance(self) -> float:
    return self._previous_feature_distance

  @property
  def norm_feature_distance(self) -> float:
    return self._previous_feature_distance / self._init_feature_distance

  @property
  def current_src(self) -> str:
    return self._previous_src


class BacktrackingModel(models.Model):
  """A CLgen model which uses a backtracking approach to sampling."""

  def __init__(self, *args, logger=None, **kwargs):
    super(BacktrackingModel, self).__init__(*args, **kwargs)
    if not isinstance(self.backend, tensorflow_backend.TensorFlowBackend):
      raise TypeError(f"{self(type).__name__} only compatible with "
                      "TensorFlow backend!")

    self._logger = logger
    self._target_features = None
    if FLAGS.experimental_clgen_backtracking_target_features:
      self._target_features = np.fromstring(
          FLAGS.experimental_clgen_backtracking_target_features,
          dtype=int,
          sep=',')
      app.Log(1, "Using target features %s", self._target_features)
      assert self._target_features.shape == (4,)

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

    backtracker = OpenClBacktrackingHelper(atomizer, self._target_features)
    self._logger.OnSampleStart(backtracker)
    sampled_tokens = self.SampleOneWithBacktracking(sampler, atomizer,
                                                    backtracker)
    self._logger.OnSampleEnd(backtracker)

    end_time = labdate.MillisecondsTimestamp()

    # Format text.
    if sampled_tokens:
      text = preprocessors.Preprocess(''.join(sampled_tokens), [
          'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
          'deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype',
          'deeplearning.clgen.preprocessors.common:StripTrailingWhitespace',
          'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
          'deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines',
          'deeplearning.clgen.preprocessors.cxx:ClangFormat',
      ])
    else:
      text = ''

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
      sampler: A Sampler instance, used to determine the start text, and when to
        terminate sampling.
      atomizer: The corpus vocabulary atomizer.
      backtracker: An instance of the backtracking helper class.

    Returns:
      A sample, as a sequence of strings.
    """
    # During sampling, 'sample_in_progress' contains the current candidate
    # sequence of tokens, and 'backtrack_state' contains the sequence of tokens
    # that is restored when backtracking.
    sample_in_progress = sampler.tokenized_start_text.copy()
    backtrack_state = sample_in_progress.copy()
    if FLAGS.experimental_clgen_backtracking_lstm_state:
      rollback_backend_state = copy.deepcopy(self.backend.inference_state)
      rollback_backend_indices = self.backend.inference_indices.copy()

    # This counter is incremented every time ShouldProceed() returns False. If
    # the value grows to exceed the
    # --experimental_clgen_backtracking_max_attempts flag, backtracking aborts.
    # The value is reset when ShouldProceed() returns True.
    backtrack_attempt_count = 0
    # This counter is incremented every time ShouldProceed() returns True. If it
    # grows to exceed the --experimental_clgen_backtracking_max_step
    step_count = 0

    while True:
      sampled_indices = self.backend.SampleNextIndices(
          sampler, batch_size=1, done=np.array([False]))[0]

      for sampled_index in sampled_indices:
        token = atomizer.decoder[sampled_index]
        sample_in_progress.append(token)

        if backtracker.ShouldCheckpoint(token):
          # There are three possible outcomes:
          #   1. Sampling should proceed.
          #   2. Sampling should terminate (because we're reached some counter
          #      limit).
          #   3. Sampling should backtrack.
          if backtracker.ShouldProceed(sample_in_progress):
            step_count += 1
            self._logger.OnSampleStep(backtracker, backtrack_attempt_count,
                                      len(sample_in_progress))
            # Set a new state to backtrack to.
            backtrack_state = sample_in_progress.copy()
            backtrack_attempt_count = 0
            if FLAGS.experimental_clgen_backtracking_lstm_state:
              rollback_backend_state = copy.deepcopy(
                  self.backend.inference_state)
              rollback_backend_indices = self.backend.inference_indices.copy()
            else:
              # Set the sampler's seed text to be the new rollback state so that
              # when InitSampleBatch() is called during backtracking (below),
              # the model is re-seeded with the entire sample up to this point.
              sampler.encoded_start_text = atomizer.AtomizeString(
                  ''.join(backtrack_state))[-(sampler.sequence_length - 1):]
          elif (
              (backtrack_attempt_count >=
               FLAGS.experimental_clgen_backtracking_max_attempts) or
              (step_count >= FLAGS.experimental_clgen_backtracking_max_steps)):
            # This branch provides a get-out in case sampling ever gets "stuck".
            # If no progress was made, backtrack_state will still be the
            # (invalid) kernel seed text, so cannot be closed.
            rollback_src = backtracker.TryToCloseProgram(backtrack_state) or ''
            return atomizer.TokenizeString(rollback_src)
          else:
            # Backtrack. Reset the backend state to the last good state.
            sample_in_progress = backtrack_state.copy()
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
        elif (sampler.SampleIsComplete(sample_in_progress) and
              backtracker.IsDone()):
          return sample_in_progress
