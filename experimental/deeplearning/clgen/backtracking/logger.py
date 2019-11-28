"""Results logging for backtracking experiments."""
import time

from experimental.deeplearning.clgen.backtracking import backtracking_db
from experimental.deeplearning.clgen.backtracking import backtracking_model
from labm8.py import app
from labm8.py import humanize

FLAGS = app.FLAGS


class BacktrackingLogger(object):

  def OnSampleStart(self,
                    backtracker: backtracking_model.OpenClBacktrackingHelper):
    pass

  def OnSampleStep(self,
                   backtracker: backtracking_model.OpenClBacktrackingHelper,
                   attempt_count: int, token_count: int):
    pass

  def OnSampleEnd(self,
                  backtracker: backtracking_model.OpenClBacktrackingHelper):
    pass


class BacktrackingDatabaseLogger(BacktrackingLogger):

  def __init__(self, db: backtracking_db.Database):
    self._db = db
    self._job_id = None
    self._step_count = 0
    self._target_features_id = None
    self._start_time = None

  def OnSampleStart(self,
                    backtracker: backtracking_model.OpenClBacktrackingHelper):
    with self._db.Session(commit=True) as session:
      target_features = session.GetOrAdd(
          backtracking_db.FeatureVector,
          **backtracking_db.FeatureVector.FromNumpyArray(
              backtracker.target_features))
      session.flush()
      self._target_features_id = target_features.id

    self._step_count = 0
    self._start_time = time.time()

  def OnSampleStep(self,
                   backtracker: backtracking_model.OpenClBacktrackingHelper,
                   attempt_count: int, token_count: int):
    job_id = self.job_id
    self._step_count += 1

    runtime_ms = int((time.time() - self._start_time) * 1000)
    app.Log(1, 'Reached step %d after %d attempts, %d tokens', self._step_count,
            attempt_count, token_count)
    app.Log(1, 'Job %d started %s', job_id,
            humanize.Duration(runtime_ms / 1000))

    with self._db.Session(commit=True) as session:
      features = session.GetOrAdd(
          backtracking_db.FeatureVector,
          **backtracking_db.FeatureVector.FromNumpyArray(
              backtracker.current_features))
      session.flush()

      step = backtracking_db.BacktrackingStep(
          job_id=job_id,
          runtime_ms=runtime_ms,
          target_features_id=self._target_features_id,
          features_id=features.id,
          feature_distance=backtracker.feature_distance,
          norm_feature_distance=backtracker.norm_feature_distance,
          step=self._step_count,
          attempt_count=attempt_count,
          src=backtracker.current_src,
          token_count=token_count,
      )
      session.add(step)

  def OnSampleEnd(self,
                  backtracker: backtracking_model.OpenClBacktrackingHelper):
    del backtracker
    self._step_count += 1
    app.Log(1, "Sampling concluded at step %d", self._step_count)
    self._job_id = None
    self._step_count = 0
    self._target_features_id = None

  @property
  def job_id(self) -> int:
    """Get the unique job ID."""
    if self._job_id is None:
      with self._db.Session(commit=True) as session:
        result = session.query(backtracking_db.BacktrackingStep.job_id).order_by(
            backtracking_db.BacktrackingStep.job_id.desc()) \
            .limit(1).first()
        if result:
          self._job_id = result[0] + 1
        else:
          self._job_id = 1
      app.Log(1, 'New job ID %d', self._job_id)
    return self._job_id
