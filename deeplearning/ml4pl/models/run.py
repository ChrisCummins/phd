"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import pyfiglet

from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batchs
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import logger as logger_lib
from deeplearning.ml4pl.models import schedules
from labm8.py import app
from labm8.py import pdutil
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
app.DEFINE_enum(
  "save_on",
  schedules.SaveOn,
  schedules.SaveOn.EVERY_EPOCH,
  "The type of checkpoints to save.",
)
app.DEFINE_enum(
  "test_on",
  schedules.TestOn,
  schedules.TestOn.IMPROVEMENT,
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
app.DEFINE_boolean(
  "k_fold",
  False,
  "If set, iterate over all K splits in the database, training and evaluating "
  "a model for each.",
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
  val_splits: Optional[List[int]] = None,
  test_splits: Optional[List[int]] = None,
  ctx: progress.ProgressContext = progress.NullContext,
) -> batchs.BatchIterator:
  """Create an iterator over batches."""
  # Filter the graph database to load graphs from the requested splits.
  val_splits = val_splits or SplitStringsToInts(FLAGS.val_split)
  test_splits = test_splits or SplitStringsToInts(FLAGS.test_split)

  if epoch_type == epoch.Type.TRAIN:
    splits = list(set(graph_db.splits) - set(val_splits) - set(test_splits))
    limit = FLAGS.max_train_per_epoch
  elif epoch_type == epoch.Type.VAL:
    splits = val_splits
    limit = FLAGS.max_val_per_epoch
  elif epoch_type == epoch.Type.TEST:
    splits = test_splits
    limit = None  # Never limit the test set.
  else:
    raise NotImplementedError("unreachable")

  ctx.Log(
    3, "Using %s graph splits %s", epoch_type.name.lower(), sorted(splits)
  )

  if len(splits) == 1:
    split_filter = lambda: graph_tuple_database.GraphTuple.split == splits[0]
  else:
    split_filter = lambda: graph_tuple_database.GraphTuple.split.in_(splits)

  graph_reader = graph_database_reader.BufferedGraphReader.CreateFromFlags(
    filters=[split_filter], ctx=ctx, limit=limit
  )

  return batchs.BatchIterator(
    batches=ppar.ThreadedIterator(
      model.BatchIterator(graph_reader, ctx=ctx),
      max_queue_size=FLAGS.batch_queue_size,
    ),
    graph_count=graph_reader.n,
  )


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
  ):
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

  def Run(self):
    """Run the train/val/test loop."""
    test_on = FLAGS.test_on()
    save_on = FLAGS.save_on()

    # Epoch loop.
    for self.ctx.i in range(self.ctx.i, self.ctx.n):
      # Create the batch iterators ahead of time so that they can asynchronously
      # start reading from the graph database.
      batch_iterators = {
        epoch_type: MakeBatchIterator(
          epoch_type, self.model, self.graph_db, ctx=self.ctx
        )
        for epoch_type in [epoch.Type.TRAIN, epoch.Type.VAL, epoch.Type.TEST]
      }

      train_results, _ = self.RunEpoch(epoch.Type.TRAIN, batch_iterators)

      val_results, val_improved = self.RunEpoch(epoch.Type.VAL, batch_iterators)

      if test_on == schedules.TestOn.IMPROVEMENT and val_improved:
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


def PrintExperimentFooter(
  model: classifier_base.ClassifierBase, logger: logger_lib.Logger
) -> pd.Series:
  epochs = GetModelEpochsTable(model, logger)
  best_epoch_num = epochs["val_accuracy"].idxmax()
  best_epoch = epochs.loc[best_epoch_num]

  print(f"\rResults at best val epoch {best_epoch_num} / {model.epoch_num}:")
  print("==================================================================")
  print(best_epoch.to_string())
  print("==================================================================")
  return best_epoch


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
  val_splits: Optional[List[int]] = None,
  test_splits: Optional[List[int]] = None,
) -> pd.Series:
  graph_db: graph_tuple_database.Database = FLAGS.graph_db()
  with logger_lib.Logger.FromFlags() as logger:
    model = CreateModel(model_class, graph_db, logger)

    PrintExperimentHeader(model)

    if FLAGS.test_only:
      batch_iterator = MakeBatchIterator(
        epoch_type=epoch.Type.TEST,
        model=model,
        graph_db=graph_db,
        val_splits=val_splits,
        test_splits=test_splits,
      )
      RunEpoch(
        epoch_name="test",
        model=model,
        batch_iterator=batch_iterator,
        epoch_type=epoch.Type.TEST,
        logger=logger,
      )
    else:
      train = Train(model, graph_db, logger)
      progress.Run(train)
      if train.ctx.i != train.ctx.n:
        app.FatalWithoutStackTrace("Model failed")

    return PrintExperimentFooter(model, logger)


def RunKFold(model_class):
  graph_db: graph_tuple_database.Database = FLAGS.graph_db()

  splits: List[int] = graph_db.splits
  results: List[pd.Series] = []
  for i in range(len(splits)):
    test_split = splits[i]
    val_split = splits[(i + 1) % len(splits)]
    results.append(
      RunOne(model_class, val_splits=[val_split], test_splits=[test_split])
    )

  # Concatenate each of the run results into a dataframe.
  df = pd.concat(results, axis=1).transpose()
  # Get the list of run names and remove them from the columns. We'll set them
  # again later.
  df.set_index("run_id", inplace=True)
  run_ids = list(df.index.values)
  # Select only the subset of columns that we're interested in: test metrics.
  df = df[
    ["test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"]
  ]
  # Add an averages row.
  df = df.append(df.mean(axis=0), ignore_index=True)
  # Strip the "test_" prefix from column names.
  df.rename(
    columns={c: c[len("test_") :] for c in df.columns.values}, inplace=True
  )
  # Set the run IDs again.
  df["run_id"] = run_ids + ["Average"]
  df.set_index("run_id", inplace=True)
  print()
  print(f"Tests results of {len(results)}-fold cross-validation:")
  print(pdutil.FormatDataFrameAsAsciiTable(df))

  return df


def Run(model_class) -> Union[pd.Series, pd.DataFrame]:
  """Run the model."""
  if FLAGS.k_fold:
    return RunKFold(model_class)
  else:
    return RunOne(model_class)


def Main():
  Run(classifier_base.ClassifierBase)


if __name__ == "__main__":
  app.Run(Main)
