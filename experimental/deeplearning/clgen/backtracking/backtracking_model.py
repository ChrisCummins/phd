"""A CLgen model with backtracking inference."""
import collections
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

app.DEFINE_integer(
    'experimental_clgen_backtracking_max_attempts', 1000,
    'The maximum number of attempts to generate a single '
    'step candidate.')
app.DEFINE_integer('experimental_clgen_backtracking_candidates_per_step', 10,
                   'The number of candidates to produce for each step.')
app.DEFINE_integer(
    'experimental_clgen_backtracking_max_steps', 10000,
    'The maximum number of steps to make when generating a program.')
app.DEFINE_float(
    'experimental_clgen_backtracking_reject_no_progress_probability', 0.5,
    'The probability that a step which does not improve feature distance is '
    'rejected. A higher value means that a larger fraction of steps must '
    'directly contribute towards an improved feature distance.')
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


def SummarizeFloats(floats: typing.Iterable[float], nplaces: int = 2) -> str:
  """Summarize a sequence of floats."""
  arr = np.array(list(floats), dtype=np.float32)
  percs = " ".join([
      f'{p}%={np.percentile(arr, p):.{nplaces}f}' for p in [0, 50, 95, 99, 100]
  ])
  return (f"n={len(arr)}, mean={arr.mean():.{nplaces}f}, "
          f"stdev={arr.std():.{nplaces}f}, percentiles=[{percs}]")


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
    self._previous_src = ''
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

  def ShouldProceed(self,
                    sample_in_progress: typing.List[str],
                    force: bool = False) -> bool:
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
      app.Log(2, 'Features: %s, distance=%f, norm=%f, delta=%f', features,
              new_feature_distance,
              new_feature_distance / self._init_feature_distance,
              new_feature_distance - self._previous_feature_distance)
      if not force and new_feature_distance > self._previous_feature_distance:
        # This will only happen once feature values are great than target
        # feature values.
        app.Log(2, "Rejecting candidate because of positive feature delta")
        return False
      if (not force and
          new_feature_distance == self._previous_feature_distance and
          random.random() >
          FLAGS.experimental_clgen_backtracking_reject_no_progress_probability):
        app.Log(2, "Randomly rejecting candidate with no progress")
        return False
      self._previous_features = features
      self._previous_src = candidate_src
      self._previous_feature_distance = new_feature_distance

    return True

  def TryToExtractFeatures(self,
                           path: pathlib.Path) -> typing.Optional[np.array]:
    """Try and extract a feature vector for the program at the given path."""
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

  def IsDone(self, sample_in_progress: typing.List[str]):
    """Return whether sampling is done."""
    if self._target_features is None:
      return True
    else:
      assert self.ShouldProceed(sample_in_progress)
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

  @feature_distance.setter
  def feature_distance(self, feature_distance: float) -> float:
    self._previous_feature_distance = feature_distance

  @property
  def norm_feature_distance(self) -> float:
    return self._previous_feature_distance / self._init_feature_distance

  @property
  def current_src(self) -> str:
    return self._previous_src


# A candidate statement for an in-progress synthesized program.
CandidateStatement = collections.namedtuple('CandidateStatement',
                                            ['statement', 'feature_distance'])


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

    # We're use the sampler.encoded_start_text attribute as a way to re-seed the
    # model state during rollback, so save the original value here so that we
    # can restore it at the end of the sample batch.
    original_sampler_encoded_start_text = sampler.encoded_start_text.copy()

    self.backend.InitSampleBatch(sampler)

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

  def GenerateCandidateStatements(
      self, initial_text, initial_state, initial_index,
      backtracker: OpenClBacktrackingHelper, sampler: samplers.Sampler,
      atomizer: atomizers.AtomizerBase) -> CandidateStatement:
    """Generate a candidate statement for the current sample in progress.

    Args:
      initial_text: The previously generated and accepted src
      initial_state: The initial state of the sampler 
      initial_index: The index fed to the sampler
      backtracker: A backtracking helper instance.
      sampler: A sampler.
      atomizer: An atomizer.

    Returns:
      A tuple candidate sequence of sampled tokens, and the feature distance of
      this candidate.
    """
    init_feature_distance = backtracker.feature_distance
    self.backend.ResetSampleState(sampler,
        state=initial_state, seed=initial_index)
    candidate_statements = []

    # Set sampler state to the last good state.
    app.Log(3, 'Beginning Statement Generation attempt')
    sampled_indices = self.backend.SampleNextIndices(
          sampler, done=np.array([False]))
    
    for i, sample in enumerate(sampled_indices):
      app.Log(3, 'Parsing Batch Item %d', i)
      candidate_statement = []
      for j, sampled_index in enumerate(sample):
        token = atomizer.decoder[sampled_index]
        candidate_statement.append(token)

        if backtracker.ShouldCheckpoint(token):
          app.Log(3, 'Reached checkpoint after %d tokens', j + 1)
          # There are two possible outcomes:
          #   1. Sampling should proceed.
          #   3. Sampling should backtrack.
          if backtracker.ShouldProceed(initial_text + candidate_statement):
            app.Log(4, 'Produced candidate statement: `%s`',
                    ''.join(candidate_statement))
            # Reset feature distance.
            new_feature_distance = backtracker.feature_distance
            backtracker.feature_distance = init_feature_distance
            candidate_statements.append(CandidateStatement(
                statement=candidate_statement,
                feature_distance=new_feature_distance))
            break;
          else:
            app.Log(4, 'Rejecting candidate statement: `%s`',
                    ''.join(candidate_statement))
            break

    if len(candidate_statements) == 0:
      # Reached the end of the sample batch without generating a valid statement.
      app.Log(3, "Didn't reach checkpoint after %d tokens", len(sampled_indices[0]))
      return [CandidateStatement(statement=[], feature_distance=np.inf)]
    else:
      return candidate_statements

  def MakeProgram(self, sampled_tokens: typing.List[str],
                  backtracker: OpenClBacktrackingHelper,
                  atomizer: atomizers.AtomizerBase) -> typing.List[str]:
    """Produce a kernel from a sample."""
    src = backtracker.TryToCloseProgram(sampled_tokens) or ''
    return atomizer.AtomizeString(src)

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
    # During sampling, 'sample_in_progress' contains the sequence of tokens that
    # is restored when backtracking.
    sample_in_progress = sampler.tokenized_start_text.copy()
    rollback_state, rollback_index = self.backend.EvaluateSampleState(sampler)

    for step_count in range(
        1, FLAGS.experimental_clgen_backtracking_max_steps + 1):
      self._logger.OnSampleStep(backtracker, 0, len(sample_in_progress))
      app.Log(4, 'Current sample in progress: `%s`',
              ''.join(sample_in_progress))

      # Generate a batch of candidates and select the best.
      candidate_statements = []
      while len(candidate_statements) < \
              FLAGS.experimental_clgen_backtracking_candidates_per_step:
        candidate_statements.extend(
          self.GenerateCandidateStatements(sample_in_progress,
              rollback_state, rollback_index,
              backtracker, sampler, atomizer)) 

      best_candidate = min(
          candidate_statements, key=lambda x: x.feature_distance)
      app.Log(
          2,
          'Selected best feature distance (%f) at step %d from candidates: %s',
          best_candidate.feature_distance, step_count,
          SummarizeFloats(c.feature_distance for c in candidate_statements))
      app.Log(4, 'Selected best statement: %s',
              ''.join(best_candidate.statement))

      # Set the sampler's rollback state to be the state produced by feeding
      # the best candidate in the input, so that future samples start from
      # the right state
      if len(best_candidate.statement) > 0:
        sample_in_progress += best_candidate.statement
        encoded_best_candidate = atomizer.AtomizeString(
            ''.join(best_candidate.statement))
        arr = np.concatenate([rollback_index, encoded_best_candidate])
        self.backend.ResetSampleState(sampler, state=rollback_state, seed=arr)
        rollback_state, rollback_index = self.backend.EvaluateSampleState(sampler)

      app.Log(5, 'Current sample at step %d: %s', step_count,
              ''.join(sample_in_progress))

      if backtracker.IsDone(sample_in_progress):
        app.Log(2, 'Backtracking complete after %d steps', step_count)
        break
    else:
      app.Log(2, 'Backtracking failed to complete after %d steps', step_count)

    return self.MakeProgram(sample_in_progress, backtracker, atomizer)
