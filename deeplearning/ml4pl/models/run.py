"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Callable
from typing import List

from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import logger as logger_lib
from labm8.py import app
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


def SplitStringsToInts(split_strings: List[str]):
  """Convert string split names to integers."""
  try:
    return [int(split) for split in split_strings]
  except Exception:
    raise app.UsageError(
      f"Splits must be a list of integers, found {split_strings}"
    )


def _RunEpoch(
  epoch_name: str,
  model: classifier_base.ClassifierBase,
  graph_db: graph_tuple_database.Database,
  epoch_type: epoch.Type,
  logger: logger_lib.Logger,
  ctx: progress.Progress = progress.NullContext,
) -> epoch.Results:
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
    if results >= model.best_val_results:
      color = shell.ShellEscapeCodes.GREEN
    else:
      color = shell.ShellEscapeCodes.RED

    return (
      f"{model.run_id} {epoch_name} "
      f"{shell.ShellEscapeCodes.BOLD}{color}{results}{shell.ShellEscapeCodes.END}"
    )

  with prof.Profile(lambda t: _EpochLabel(results), print_to=ctx.print):
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

    results = model(epoch_type, graph_reader, logger)

  return results


class Train(progress.Progress):
  """The training job."""

  def __init__(
    self,
    model: classifier_base.ClassifierBase,
    graph_db: graph_tuple_database.Database,
    logger: logger_lib.Logger,
  ):
    if FLAGS.test_on not in {"none", "every", "improvement", "end"}:
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

  def Run(self) -> epoch.Results:
    """Run the train/val/test loop."""
    # Set later.
    test_results = epoch.Results.NullResults()

    for self.ctx.i in range(self.ctx.i + 1, self.ctx.n + 1):
      self.model.epoch_num = self.ctx.i
      self._RunEpoch(epoch.Type.TRAIN)
      val_results = self._RunEpoch(epoch.Type.VAL)

      if val_results > self.model.best_val_results:
        self.ctx.Log(
          2,
          "Validation results improved:\n  from: %s\n    to: %s",
          self.model.best_val_results,
          val_results,
        )
        self.model.best_val_results = val_results
        self.logger.Save(self.model.run_id, self.model.ModelDataToSave())

        if FLAGS.test_on == "improvement":
          test_results = self._RunEpoch(epoch.Type.TEST)

      if FLAGS.test_on == "every":
        test_results = self._RunEpoch(epoch.Type.TEST)

    # We have reached the end of training. If we haven't been doing incremental
    # testing, then run the test set.
    if FLAGS.test_on == "end":
      self.model.LoadModelData(self.logger.Load(FLAGS.restore_model))
      test_results = self._RunEpoch(epoch.Type.TEST)

    return test_results

  def _RunEpoch(self, epoch_type: epoch.Type):
    """Run an epoch of the given type."""
    epoch_name = (
      f"{epoch_type.name.lower():>5} " f"[{self.ctx.i:3d} / {self.ctx.n:3d}]"
    )
    return _RunEpoch(
      epoch_name, self.model, self.graph_db, epoch_type, self.logger, self.ctx
    )


ModelClass = Callable[
  [graph_tuple_database.Database], classifier_base.ClassifierBase
]


def Run(model_class: ModelClass):
  graph_db: graph_tuple_database.Database = FLAGS.graph_db()
  logger = logger_lib.Logger.FromFlags()

  # TODO(github.com/ChrisCummins/ProGraML/issues/24): Add split db.
  # split_db = FLAGS.split_db()

  # Instantiate a model.
  with prof.Profile(lambda t: f"Initialized {model.run_id}"):
    model = model_class(graph_db)
    if FLAGS.restore_model:
      # TODO(github.com/ChrisCummins/ProGraML/issues/24): Implement model
      # restoring.
      model.LoadModelData(logger.GetModelData(FLAGS.restore_model))
    else:
      model.Initialize()

  if FLAGS.test_only:
    _RunEpoch("test", model, graph_db, logger, epoch.Type.TEST)
  else:
    progress.Run(Train(model, graph_db, logger))

  print("\rdone")


def Main():
  Run(classifier_base.ClassifierBase)


if __name__ == "__main__":
  app.Run(Main)
