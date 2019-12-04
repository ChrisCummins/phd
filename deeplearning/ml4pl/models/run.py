"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batchs
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import logger as logger_lib
from labm8.py import app
from labm8.py import ppar
from labm8.py import prof
from labm8.py import progress
from labm8.py import shell


FLAGS = app.FLAGS

app.DEFINE_string(
  "restore_model",
  None,
  "Select a model checkpoint to restore the model state from. The checkpoint "
  "is identified by a run ID and optionally an epoch number, in the format "
  "--restore_model=<run_id>[:<epoch_num>]. If no epoch number is specified, "
  "the most recent epoch is used. Model checkpoints are loaded from "
  "the log database.",
)
app.DEFINE_string(
  "test_on", "improvement", "Determine when to run the test set.",
)
app.DEFINE_boolean(
  "test_only",
  False,
  "If this flag is set, only a single pass of the test set is ran.",
)
app.DEFINE_integer("epoch_count", 300, "The number of epochs to train for.")
app.DEFINE_integer(
  "patience",
  300,
  "The number of epochs to train for without any improvement in validation "
  "accuracy before stopping.",
)
app.DEFINE_integer(
  "max_train_per_epoch",
  None,
  "Use this flag to limit the maximum number of instances used in a single "
  "training epoch. For k-fold cross-validation, each of the k folds will "
  "train on a maximum of this many graphs.",
)
app.DEFINE_integer(
  "max_val_per_epoch",
  None,
  "Use this flag to limit the maximum number of instances used in a single "
  "validation epoch.",
)
app.DEFINE_list(
  "val_split",
  ["1"],
  "The names of the splits to be used for validating model performance. All "
  "splits except --val_split and --test_split will be used for training.",
)
app.DEFINE_list(
  "test_split",
  ["2"],
  "The name of the hold-out splits to be used for testing. All splits "
  "except --val_split and --test_split will be used for training.",
)
app.DEFINE_integer(
  "batch_queue_size",
  10,
  "Tuning parameter. The maximum number of batches to generate before "
  "waiting for the model to complete. Must be >= 1.",
)


def SplitStringsToInts(split_strings: List[str]):
  """Convert string split names to integers."""

  def MakeInt(split: str):
    try:
      return int(split)
    except Exception:
      raise app.UsageError(
        f"Splits must be a list of integers, found '{split}'"
      )

  return [MakeInt(split) for split in split_strings]


def MakeBatchIterator(
  epoch_type: epoch.Type,
  model: classifier_base.ClassifierBase,
  graph_db: graph_tuple_database.Database,
  ctx: progress.ProgressContext = progress.NullContext,
) -> batchs.BatchIterator:
  """Create an iterator over batches."""
  # Filter the graph database to load graphs from the requested splits.
  val_splits = SplitStringsToInts(FLAGS.val_split)
  test_splits = SplitStringsToInts(FLAGS.test_split)

  if epoch_type == epoch.Type.TRAIN:
    splits = list(set(graph_db.splits) - set(val_splits) - set(test_splits))
  elif epoch_type == epoch.Type.VAL:
    splits = val_splits
  elif epoch_type == epoch.Type.TEST:
    splits = test_splits
  ctx.Log(
    3, "Using %s graph splits %s", epoch_type.name.lower(), sorted(splits)
  )

  if len(splits) == 1:
    split_filter = lambda: graph_tuple_database.GraphTuple.split == splits[0]
  else:
    split_filter = lambda: graph_tuple_database.GraphTuple.split.in_(splits)

  graph_reader = graph_database_reader.BufferedGraphReader.CreateFromFlags(
    filters=[split_filter], ctx=ctx
  )

  return batchs.BatchIterator(
    batches=ppar.ThreadedIterator(
      model.BatchIterator(graph_reader), max_queue_size=FLAGS.batch_queue_size
    ),
    graph_count=graph_reader.n,
  )


def _RunEpoch(
  epoch_name: str,
  model: classifier_base.ClassifierBase,
  batch_iterator: batchs.BatchIterator,
  epoch_type: epoch.Type,
  logger: logger_lib.Logger,
  ctx: progress.Progress = progress.NullContext,
) -> Tuple[epoch.Results, int]:
  """

  Args:
    model: A model.
    graph_db: A graph database.
    epoch_type: The epoch type to run.
    ctx: A progress context.

  Returns:
    An epoch Results instance.
  """

  def _EpochLabel(results: epoch.Results):
    return (
      f"{shell.ShellEscapeCodes.BLUE}{model.run_id}{shell.ShellEscapeCodes.END} "
      f"{epoch_name} "
      f"{results.ToFormattedString(model.best_results[epoch_type].results)}"
    )

  with prof.Profile(lambda t: _EpochLabel(results), print_to=ctx.print):
    results = model(epoch_type, batch_iterator, logger)

  improved = model.UpdateBestResults(
    epoch_type, model.epoch_num, results, ctx=ctx
  )

  return results, improved


class Train(progress.Progress):
  """The training job."""

  def __init__(
    self,
    model: classifier_base.ClassifierBase,
    graph_db: graph_tuple_database.Database,
    logger: logger_lib.Logger,
  ):
    if FLAGS.test_on not in {"none", "improvement", "every"}:
      raise app.UsageError("Unknown --test_on option")

    self.model = model
    self.graph_db = graph_db
    self.logger = logger
    super(Train, self).__init__(
      str(self.model.run_id),
      i=self.model.epoch_num,
      n=FLAGS.epoch_count,
      unit="epoch",
      vertical_position=1,
      leave=False,
    )
    self.logger.ctx = self.ctx

  def Run(self) -> epoch.Results:
    """Run the train/val/test loop."""
    # Set later.
    test_results = epoch.Results()

    # Epoch loop.
    for self.ctx.i in range(self.ctx.i + 1, self.ctx.n + 1):
      # Create the batch iterators ahead of time so that they can asynchronously
      # start reading from the graph database.
      batch_iterators = {
        epoch_type: MakeBatchIterator(
          epoch_type, self.model, self.graph_db, self.ctx
        )
        for epoch_type in [epoch.Type.TRAIN, epoch.Type.VAL, epoch.Type.TEST]
      }

      self.model.epoch_num = self.ctx.i
      train_results, _ = self._RunEpoch(epoch.Type.TRAIN, batch_iterators)
      self.model.UpdateBestResults(
        epoch.Type.TRAIN, self.model.epoch_num, train_results, ctx=self.ctx
      )

      val_results, val_improved = self._RunEpoch(
        epoch.Type.VAL, batch_iterators
      )

      if val_improved:
        self.logger.Save(self.model.GetCheckpoint())

        if FLAGS.test_on == "improvement":
          test_results, _ = self._RunEpoch(epoch.Type.TEST, batch_iterators)

      if FLAGS.test_on == "every":
        test_results, _ = self._RunEpoch(epoch.Type.TEST, batch_iterators)
        self.model.UpdateBestResults(
          epoch.Type.TEST, self.model.epoch_num, train_results, ctx=self.ctx
        )

    return test_results

  def _RunEpoch(
    self,
    epoch_type: epoch.Type,
    batch_iterators: Dict[epoch.Type, batchs.BatchIterator],
  ) -> Tuple[epoch.Results, int]:
    """Run an epoch of the given type."""
    epoch_name = (
      f"{epoch_type.name.lower():>5} " f"[{self.ctx.i:3d} / {self.ctx.n:3d}]"
    )
    return _RunEpoch(
      epoch_name=epoch_name,
      model=self.model,
      batch_iterator=batch_iterators[epoch_type],
      epoch_type=epoch_type,
      logger=self.logger,
      ctx=self.ctx,
    )


# Type annotation for a classifier_base.ClassifierBase subclass. Note this is
# the type itself, not an instance of that type.
ModelClass = Callable[
  [graph_tuple_database.Database], classifier_base.ClassifierBase
]


def _RunWithLogger(
  model_class: ModelClass,
  graph_db: graph_tuple_database.Database,
  logger: logger_lib.Logger,
):
  # Instantiate a model.
  with prof.Profile(
    lambda t: f"Initialized {model.run_id}",
    print_to=lambda msg: app.Log(2, msg),
  ):
    if FLAGS.restore_model:
      model_class.FromCheckpoint(
        logger.Load(
          *checkpoints.RunIdAndEpochNumFromString(FLAGS.restore_model)
        )
      )
    else:
      model = model_class(
        node_y_dimensionality=graph_db.node_y_dimensionality,
        graph_y_dimensionality=graph_db.graph_y_dimensionality,
      )
      model.Initialize()

  logger.OnStartRun(model.run_id)

  if FLAGS.test_only:
    batch_iterator = MakeBatchIterator(
      epoch_type=epoch.Type.TEST, model=model, graph_db=graph_db
    )
    _RunEpoch(
      epoch_name="test",
      model=model,
      batch_iterator=batch_iterator,
      epoch_type=epoch.Type.TEST,
      logger=logger,
    )
  else:
    progress.Run(Train(model, graph_db, logger))

  print("\rdone")


def Run(model_class: ModelClass):
  graph_db: graph_tuple_database.Database = FLAGS.graph_db()
  with logger_lib.Logger.FromFlags() as logger:
    _RunWithLogger(model_class, graph_db, logger)


def Main():
  Run(classifier_base.ClassifierBase)


if __name__ == "__main__":
  app.Run(Main)
