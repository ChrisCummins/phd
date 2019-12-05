"""This module defines a generator for databases of random model logs.

When executed as a script, this generates and populates a database of logs:

    $ bazel run //deeplearning/ml4pl/testing:random_log_database_generator -- \
        --log_db='sqlite:////tmp/logs.db' \
        --run_count=10
"""
import copy
import random
from typing import List
from typing import Optional

import numpy as np

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import decorators
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_integer(
  "random_pool_size", 256, "The maximum number of random rows to generate.",
)
app.DEFINE_integer(
  "run_count",
  10,
  "When //deeplearning/ml4pl/testing:random_log_database_generator is executed "
  "as a script, this determines the number of runs to populate --log_db with.",
)


class RandomLogDatabaseGenerator(object):
  """A generator for random logs."""

  def CreateRandomRunLogs(
    self,
    run_id: Optional[run_id_lib.RunId] = None,
    max_param_count: Optional[int] = None,
    max_epoch_count: Optional[int] = None,
    max_batch_count: Optional[int] = None,
  ) -> log_database.RunLogs:
    """Create random logs for a run.

    Args:
      run_id: The run to generate parameters for.
      max_param_count: The maximum number of parameters to generate.
      max_epoch_count: The maximum number of epochs to generate logs for.
      max_batch_count: The maximum number of batches to generate for an epoch of
        a given type.
    """
    run_id = log_database.RunId(
      run_id=str(
        run_id
        or run_id_lib.RunId.GenerateUnique(
          f"rand{random.randint(0, 10000):04d}"
        )
      )
    )
    parameters = self._CreateRandomParameters(
      run_id, max_param_count=max_param_count
    )
    batches = self._CreateRandomBatches(
      run_id, max_epoch_count=max_epoch_count, max_batch_count=max_batch_count
    )
    # TODO: Generate checkpoints
    return log_database.RunLogs(
      run_id=run_id, parameters=parameters, batches=batches, checkpoints=[]
    )

  def PopulateLogDatabase(
    self,
    db: log_database.Database,
    run_count: int,
    max_param_count: Optional[int] = None,
    max_epoch_count: Optional[int] = None,
    max_batch_count: Optional[int] = None,
  ):
    for i in range(run_count):
      with db.Session(commit=True) as session:
        session.add_all(
          self.CreateRandomRunLogs(
            run_id=run_id_lib.RunId.GenerateUnique(f"rand{i:04d}"),
            max_param_count=max_param_count,
            max_epoch_count=max_epoch_count,
            max_batch_count=max_batch_count,
          ).all
        )

  #############################################################################
  # Private members.
  #############################################################################

  def _CreateRandomParameters(
    self, run_id: log_database.RunId, max_param_count: Optional[int] = None
  ) -> List[log_database.Parameter]:
    """Generate random parameter logs.

    Args:
      run_id: The run to generate parameters for. This is assumed to be unique.
      max_param_count: The maximum number of parameters to generate.
    """
    param_count = min(
      max_param_count or random.randint(20, 50), len(self.random_parameters)
    )
    params = [
      copy.deepcopy(self.random_parameters[i]) for i in range(param_count)
    ]
    for param in params:
      param.run_id = run_id
    return params

  def _CreateRandomBatches(
    self,
    run_id: log_database.RunId,
    max_epoch_count: Optional[int] = None,
    max_batch_count: Optional[int] = None,
  ) -> List[log_database.Batch]:
    """Create random batch logs for the given run.

    Args:
      run_id: The run to generate batches for. This is assumed to be unique.
      max_epoch_count: The maximum number of epochs to generate logs for.
      max_batch_count: The maximum number of batches to generate for an epoch of
        a given type.
    """
    batches = []
    epoch_count = random.randint(1, max_epoch_count or 10)
    # Use the same number of batches for each of epoch of a given type.
    batch_counts = {
      epoch.Type.TRAIN: random.randint(1, max_batch_count or 100),
      epoch.Type.VAL: random.randint(1, max_batch_count or 30),
      epoch.Type.TEST: random.randint(1, max_batch_count or 30),
    }

    # Generate per-epoch logs.
    for epoch_num in range(1, epoch_count + 1):
      epoch_types = [epoch.Type.TRAIN, epoch.Type.VAL]
      # Randomly determine whether to include testing logs.
      if random.random() < 0.3:
        epoch_types.append(epoch.Type.TEST)

      for epoch_type in epoch_types:
        batch_count = batch_counts[epoch_type]
        epoch_batches = [
          copy.deepcopy(random.choice(self.random_batches))
          for _ in range(batch_count)
        ]
        for i, batch in enumerate(epoch_batches):
          batch.run_id = run_id
          batch.epoch_num = epoch_num
          batch.epoch_type = epoch_type
          batch.batch_num = i + 1
        batches += epoch_batches

    return batches

  @decorators.memoized_property
  def random_parameters(self) -> List[log_database.Parameter]:
    """Get a pool of random parameters.

    Calling code must set the run_id value.
    """
    return [
      log_database.Parameter.Create(
        run_id=run_id_lib.RUN_ID,
        type=random.choice(list(log_database.ParameterType)),
        name=f"param_{i:03d}",
        value=random.random(),
      )
      for i in range(FLAGS.random_pool_size)
    ]

  @decorators.memoized_property
  def random_batches(self) -> List[log_database.Batch]:
    """Get a pool of random batch logs.

    Calling code must set the run_id, epoch_type, epoch_num, and batch_num
    values.
    """
    batches = []

    for batch_num in range(1, FLAGS.random_pool_size):
      data = random.choice(self.random_batch_data)
      results = random.choice(self.random_batch_results)
      batch = log_database.Batch.Create(
        run_id=run_id_lib.RUN_ID,
        epoch_type=epoch.Type.TRAIN,
        epoch_num=0,
        batch_num=batch_num,
        timer=prof.ProfileTimer(),
        data=data,
        results=results,
      )
      if random.random() < 0.5:
        batch.details = log_database.BatchDetails.Create(
          data=data, results=results,
        )
      batches.append(batch)
    return batches

  @decorators.memoized_property
  def random_batch_data(self) -> List[batches.Data]:
    """Get a pool of random batch data."""
    return [
      batches.Data(
        graph_ids=list(range(random.randint(10, 1000))),
        data=list(range(random.randint(10000, 100000))),
      )
      for _ in range(FLAGS.random_pool_size)
    ]

  @decorators.memoized_property
  def random_batch_results(self) -> List[batches.Results]:
    """Get a pool of random batch results."""
    results = []
    for i in range(FLAGS.random_pool_size):
      target_count = random.randint(10, 1000)
      results.append(
        batches.Results.Create(
          targets=np.random.rand(target_count, 5),
          predictions=np.random.rand(target_count, 5),
          iteration_count=random.randint(1, 3),
          model_converged=random.choice([False, True]),
          loss=random.random(),
        )
      )
    return results


def Main():
  """Main entry point"""
  log_db = FLAGS.log_db()

  log_db_generator = RandomLogDatabaseGenerator()
  log_db_generator.PopulateLogDatabase(log_db, run_count=FLAGS.run_count)


if __name__ == "__main__":
  app.Run(Main)
