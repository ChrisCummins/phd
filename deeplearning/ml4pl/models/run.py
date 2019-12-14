"""Run script for machine learning models.

This defines the schedules for running training / validation / testing loops
of a machine learning model.
"""
import warnings
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import memory_profiler
import pandas as pd
import pyfiglet
from sklearn.exceptions import UndefinedMetricWarning

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batchs
from deeplearning.ml4pl.models import batch_iterator as batch_iterator_lib
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import logger as logger_lib
from deeplearning.ml4pl.models import schedules
from labm8.py import app
from labm8.py import pdutil
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
app.DEFINE_enum(
  "save_on",
  schedules.SaveOn,
  schedules.SaveOn.EVERY_EPOCH,
  "The type of checkpoints to save.",
)
app.DEFINE_enum(
  "test_on",
  schedules.TestOn,
  schedules.TestOn.IMPROVEMENT_AND_LAST,
  "Determine when to run the test set.",
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
app.DEFINE_boolean(
  "k_fold",
  False,
  "If set, iterate over all K splits in the database, training and evaluating "
  "a model for each.",
)
app.DEFINE_boolean(
  "run_with_memory_profiler",
  False,
  "If set, run the program with memory profiling enabled. See "
  "https://pypi.org/project/memory-profiler/",
)


class RunError(OSError):
  """An error that occurs during model execution."""

  pass


def SplitStringsToInts(split_strings: List[str]):
  """Convert string split names to integers."""

  def MakeInt(split: str):
    try:
      return int(split)
    except Exception:
      raise app.UsageError(f"Invalid split number: {split}")

  return [MakeInt(split) for split in split_strings]


def SplitsFromFlags(
  graph_db: graph_tuple_database.Database,
) -> Dict[epoch.Type, List[int]]:
  val_splits = SplitStringsToInts(FLAGS.val_split)
  test_splits = SplitStringsToInts(FLAGS.test_split)
  train_splits = list(set(graph_db.splits) - set(val_splits) - set(test_splits))
  return {
    epoch.Type.TRAIN: train_splits,
    epoch.Type.VAL: val_splits,
    epoch.Type.TEST: test_splits,
  }


def RunEpoch(
  epoch_name: str,
  model: classifier_base.ClassifierBase,
  batch_iterator: batchs.BatchIterator,
  epoch_type: epoch.Type,
  logger: logger_lib.Logger,
) -> Tuple[epoch.Results, int]:
  """Run a single epoch.

  Args:
    epoch_name: The name of the epoch, used to generate the logging output.
    epoch_type: The type of epoch to run.
    model: The model to run an epoch on.
    batch_iterator: An iterator over batches.
    logger: A logger instance.
    ctx: A progress context.

  Returns:
    The results of the epoch.
  """

  def GetEpochLabel(results: epoch.Results) -> str:
    """Generate the label for an epoch. This is printed to stdout."""
    return (
      f"{shell.ShellEscapeCodes.BLUE}{model.run_id}{shell.ShellEscapeCodes.END} "
      f"{epoch_name} "
      f"{results.ToFormattedString(model.best_results[epoch_type].results)}"
    )

  with prof.Profile(
    lambda t: GetEpochLabel(results), print_to=logger.ctx.print
  ):
    results = model(epoch_type, batch_iterator, logger)

  # Check if the model has improved.
  improved = model.best_results[epoch_type].epoch_num == model.epoch_num

  return results, improved


class Train(progress.Progress):
  """A training job. This implements train/val/test schedule."""

  def __init__(
    self,
    model: classifier_base.ClassifierBase,
    graph_db: graph_tuple_database.Database,
    logger: logger_lib.Logger,
    splits: Dict[epoch.Type, List[int]],
  ):
    self.model = model
    self.graph_db = graph_db
    self.logger = logger
    self.splits = splits
    super(Train, self).__init__(
      str(self.model.run_id),
      i=self.model.epoch_num,
      n=FLAGS.epoch_count,
      unit="epoch",
      vertical_position=1,
      leave=False,
    )
    self.logger.ctx = self.ctx

  def Run(self):
    """Run the train/val/test loop."""
    test_on = FLAGS.test_on()
    save_on = FLAGS.save_on()

    # Epoch loop.
    for self.ctx.i in range(self.ctx.i, self.ctx.n):
      # Create the batch iterators ahead of time so that they can asynchronously
      # start reading from the graph database.
      batch_iterators = {
        epoch_type: batch_iterator_lib.MakeBatchIterator(
          model=self.model,
          graph_db=self.graph_db,
          splits=self.splits,
          epoch_type=epoch_type,
          ctx=self.ctx,
        )
        for epoch_type in [epoch.Type.TRAIN, epoch.Type.VAL, epoch.Type.TEST]
      }

      train_results, _ = self.RunEpoch(epoch.Type.TRAIN, batch_iterators)

      val_results, val_improved = self.RunEpoch(epoch.Type.VAL, batch_iterators)

      if val_improved and (
        test_on == schedules.TestOn.IMPROVEMENT
        or test_on == schedules.TestOn.IMPROVEMENT_AND_LAST
      ):
        self.RunEpoch(epoch.Type.TEST, batch_iterators)
      elif (
        test_on == schedules.TestOn.IMPROVEMENT_AND_LAST
        and self.ctx.i == self.ctx.n - 1
      ):
        self.RunEpoch(epoch.Type.TEST, batch_iterators)

      # Determine whether to make a checkpoint.
      if save_on == schedules.SaveOn.EVERY_EPOCH or (
        save_on == schedules.SaveOn.VAL_IMPROVED and val_improved
      ):
        self.model.SaveCheckpoint()

      if test_on == schedules.TestOn.EVERY:
        self.RunEpoch(epoch.Type.TEST, batch_iterators)

    # Record the final epoch.
    self.ctx.i += 1

  def RunEpoch(
    self,
    epoch_type: epoch.Type,
    batch_iterators: Dict[epoch.Type, batchs.BatchIterator],
  ) -> Tuple[epoch.Results, int]:
    """Run an epoch of the given type."""
    epoch_name = (
      f"{epoch_type.name.lower():>5} "
      f"[{self.model.epoch_num:3d} / {self.ctx.n:3d}]"
    )
    return RunEpoch(
      epoch_name=epoch_name,
      model=self.model,
      batch_iterator=batch_iterators[epoch_type],
      epoch_type=epoch_type,
      logger=self.logger,
    )


def PrintExperimentHeader(model: classifier_base.ClassifierBase) -> None:
  print("==================================================================")
  print(pyfiglet.figlet_format(model.run_id.script_name))
  print("Run ID:", model.run_id)
  params = model.parameters[["type", "name", "value"]]
  params = params.rename(columns=({"type": "parameter"}))
  print(pdutil.FormatDataFrameAsAsciiTable(params))
  print()
  print(model.Summary())
  print("==================================================================")


def PrintExperimentFooter(
  model: classifier_base.ClassifierBase, best_epoch: pd.Series
) -> None:
  print(
    f"\rResults at best val epoch {best_epoch['epoch_num']} / {model.epoch_num}:"
  )
  print("==================================================================")
  print(best_epoch.to_string())
  print("==================================================================")


def GetModelEpochsTable(
  model: classifier_base.ClassifierBase, logger: logger_lib.Logger
) -> pd.DataFrame:
  """Compute a table of per-epoch stats for the model."""
  with logger.Session() as session:
    for name, df in logger.db.GetTables(
      run_ids=[model.run_id], session=session
    ):
      if name == "epochs":
        return df.set_index("epoch_num")


def CreateModel(
  model_class, graph_db, logger
) -> classifier_base.ClassifierBase:
  model: classifier_base.ClassifierBase = model_class(
    logger=logger, graph_db=graph_db
  )

  if FLAGS.restore_model:
    with prof.Profile(
      lambda t: f"Restored {model.run_id} from {checkpoint_ref}",
      print_to=lambda msg: app.Log(2, msg),
    ):
      checkpoint_ref = checkpoints.CheckpointReference.FromString(
        FLAGS.restore_model
      )
      model.RestoreFrom(checkpoint_ref)
  else:
    with prof.Profile(
      lambda t: f"Initialized {model.run_id}",
      print_to=lambda msg: app.Log(2, msg),
    ):
      model.Initialize()

  return model


def RunOne(
  model_class,
  print_header: bool = True,
  print_footer: bool = True,
  ctx: progress.ProgressContext = progress.NullContext,
) -> pd.Series:
  graph_db: graph_tuple_database.Database = FLAGS.graph_db()
  with logger_lib.Logger.FromFlags() as logger:
    logger.ctx = ctx
    model = CreateModel(model_class, graph_db, logger)

    if print_header:
      PrintExperimentHeader(model)

    splits = SplitsFromFlags(graph_db)

    if FLAGS.test_only:
      batch_iterator = batch_iterator_lib.MakeBatchIterator(
        model=model,
        graph_db=graph_db,
        splits=splits,
        epoch_type=epoch.Type.TEST,
      )
      RunEpoch(
        epoch_name="test",
        model=model,
        batch_iterator=batch_iterator,
        epoch_type=epoch.Type.TEST,
        logger=logger,
      )
    else:
      train = Train(
        model=model, graph_db=graph_db, logger=logger, splits=splits
      )
      progress.Run(train)
      if train.ctx.i != train.ctx.n:
        raise RunError("Model failed")

    # Get the results for the best epoch.
    epochs = GetModelEpochsTable(model, logger)

    # Select only from the epochs with test accuracy, if available.
    only_with_test_epochs = epochs[
      (epochs["test_accuracy"].astype(str) != "-")
      & (epochs["test_accuracy"].notnull())
    ]
    if len(only_with_test_epochs):
      epochs = only_with_test_epochs

    epochs.reset_index(inplace=True)

    # Select the row with the greatest validation accuracy.
    best_epoch = epochs.loc[epochs["val_accuracy"].idxmax()]

    if print_footer:
      PrintExperimentFooter(model, best_epoch)

    return best_epoch


class KFoldCrossValidation(progress.Progress):
  """A k cross-validation jobs.

  This runs the requested train/val/test schedule using every split in the
  graph database.
  """

  def __init__(
    self, model_class,
  ):
    """Constructor.

    Args:
      model_class: A model constructor.

    Raises:
      ValueError: If the database contains invalid splits.
    """
    self.model_class = model_class
    self.graph_db: graph_tuple_database.Database = FLAGS.graph_db()
    self.results: Optional[pd.DataFrame] = []
    if not self.graph_db.splits:
      raise ValueError("Database contains no splits")
    if self.graph_db.splits != list(range(len(self.graph_db.splits))):
      raise ValueError(
        "Graph database splits are not a contiguous sequence: "
        f"{self.graph_db.splits}"
      )
    super(KFoldCrossValidation, self).__init__(
      f"{self.graph_db.split_count}-fold xval",
      i=0,
      n=self.graph_db.split_count,
      unit="split",
      # Stack below the per-epoch and per-model progress bars.
      vertical_position=2,
      leave=False,
    )

  def Run(self):
    """Run the train/val/test loop."""
    results: List[pd.Series] = []
    splits = self.graph_db.splits

    # This assumes splits have values [0, ..., n-1].
    for i in range(len(splits)):
      self.ctx.i = i

      # Set the splits flags so that the logger captures the correct
      # parameters.
      FLAGS.test_split = [str(splits[i])]
      FLAGS.val_split = [str(splits[(i + 1) % len(splits)])]

      # Print the header only on the first split.
      print_header = False if i else True

      results.append(
        RunOne(self.model_class, print_header=print_header, ctx=self.ctx,)
      )
    self.ctx.i += 1

    # If we got no results then there was an error during model runs.
    if not results:
      return

    # Concatenate each of the run results into a dataframe.
    df = pd.concat(results, axis=1).transpose()
    # Get the list of run names and remove them from the columns. We'll set them
    # again later.
    df.set_index("run_id", inplace=True)
    run_ids = list(df.index.values)
    # Select only the subset of columns that we're interested in: test metrics.
    df = df[
      [
        "epoch_num",
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
      ]
    ]
    # Add an averages row.
    df = df.append(df.mean(axis=0), ignore_index=True)
    # Strip the "test_" prefix from column names.
    df.rename(
      columns={
        c: c[len("test_") :] for c in df.columns.values if c.startswith("test_")
      },
      inplace=True,
    )
    # Set the run IDs again.
    df["run_id"] = run_ids + ["Average"]
    df.set_index("run_id", inplace=True)
    print()
    print(f"Tests results of {len(results)}-fold cross-validation:")
    print(pdutil.FormatDataFrameAsAsciiTable(df))

    self.results = df


def RunKFold(model_class) -> pd.DataFrame:
  """Run k-fold cross-validation of the given model."""
  kfold = KFoldCrossValidation(model_class)
  progress.Run(kfold)
  if kfold.ctx.i != kfold.ctx.n:
    raise RunError(
      f"Expected to run {kfold.ctx.n} folds but only ran {kfold.ctx.i}"
    )
  if not isinstance(kfold.results, pd.DataFrame):
    raise RunError("K-fold returned no results")
  if len(kfold.results) < kfold.ctx.n:
    raise RunError(
      f"Ran {kfold.ctx.n} folds buts only have {kfold.results} results"
    )
  return kfold.results


def _RunFlagsActionsOnModelOrDir(model_class):
  """Run the model with the requested flags actions.

  Args:
    model_class: The model to run.

  Returns:
    A DataFrame of k-fold results, or a single series of results.
  """
  if not FLAGS.graph_db:
    raise app.UsageError("--graph_db is required")

  # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
  # warnings that it's undefined when there are missing instances from a class,
  # which is fine for our usage.
  warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

  try:
    if FLAGS.k_fold:
      return RunKFold(model_class)
    else:
      return RunOne(model_class)
  except RunError as e:
    app.FatalWithoutStackTrace("%s", e)


memory_profiler_log = open("/tmp/memory_profiler.log", "w+")


@memory_profiler.profile(stream=memory_profiler_log)
def RunWithMemoryProfiler(func, *args, **kwargs):
  """Given the given argument with a memory profiler.

  See: https://pypi.org/project/memory-profiler/
  """
  return func(*args, **kwargs)


def Run(model_class) -> Optional[Union[pd.Series, pd.DataFrame]]:
  """Run the model."""
  if FLAGS.run_with_memory_profiler:
    return RunWithMemoryProfiler(_RunFlagsActionsOnModelOrDir, model_class)
  else:
    return _RunFlagsActionsOnModelOrDir(model_class)
