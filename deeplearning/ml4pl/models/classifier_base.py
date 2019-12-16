"""Base class for implementing classifier models."""
import copy
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set

import pandas as pd

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import logger as logging
from labm8.py import app
from labm8.py import decorators
from labm8.py import humanize
from labm8.py import progress


FLAGS = app.FLAGS


app.DEFINE_boolean(
  "strict_graph_segmentation",
  False,
  "If set, strictly enforce that graphs do not cross the "
  "{train,val,test} epoch boundaries. This is disabled by default as the "
  "performance and memory overhead may be large for big datasets.",
)


class ClassifierBase(object):
  """Abstract base class for implementing classifiers.

  Before using the model, it must be initialized bu calling Initialize(), or
  restored from a checkpoint using RestoreFrom(checkpoint).

  Subclasses must implement the following methods:
    MakeBatch()        # construct a batch from input graphs.
    RunBatch()         # run the model on the batch.
    GetModelData()     # get model data to save.
    LoadModelData()    # load model data.

  And may optionally wish to implement these additional methods:
    CreateModelData()  # initialize an untrained model.
    Summary()          # return a string model summary.
    NeedsGraphTuples() # return whether MakeBatch() receives graph tuples.
  """

  def __init__(
    self,
    logger: logging.Logger,
    graph_db: graph_tuple_database,
    run_id: Optional[run_id_lib.RunId] = None,
  ):
    """Constructor.

    This creates an uninitialized model. Initialize the model before use by
    calling Initialize() or RestoreFrom(checkpoint).

    Args:
      logger: A logger to write {batch, epoch, checkpoint} data to.
      graph_db: The graph database which will be used to feed inputs to the
        model.

    Raises:
      NotImplementedError: If both node and graph labels are set.
      TypeError: If neither graph or node labels are set.
    """
    # Sanity check the dimensionality of input graphs.
    if (
      not graph_db.node_y_dimensionality and not graph_db.graph_y_dimensionality
    ):
      raise NotImplementedError(
        "Neither node or graph labels are set. What am I to do?"
      )
    if graph_db.node_y_dimensionality and graph_db.graph_y_dimensionality:
      raise NotImplementedError(
        "Both node and graph labels are set. This is currently not supported. "
        "See <github.com/ChrisCummins/ProGraML/issues/26>"
      )

    # Model properties.
    self.logger: logging.Logger = logger
    self.graph_db: graph_tuple_database.Database = graph_db
    self.run_id: run_id_lib.RunId = (
      run_id or run_id_lib.RunId.GenerateUnique(type(self).__name__)
    )
    self.y_dimensionality: int = (
      self.graph_db.node_y_dimensionality
      or self.graph_db.graph_y_dimensionality
    )

    # Set by Initialize() and RestoredFrom()
    self._initialized = False
    self.restored_from: Optional[checkpoints.CheckpointReference] = None

    # Progress counters that are saved and loaded from checkpoints.
    self.epoch_num = 0
    self.best_results: Dict[epoch.Type, epoch.BestResults] = {
      epoch.Type.TRAIN: epoch.BestResults(),
      epoch.Type.VAL: epoch.BestResults(),
      epoch.Type.TEST: epoch.BestResults(),
    }

    # If --strict_graph_segmentation is set, check for graphs that we have
    # already seen before by keep a log of all unique graph IDs of each type.
    self.graph_ids: Dict[epoch.Type, Set[int]] = {
      epoch.Type.TRAIN: set(),
      epoch.Type.VAL: set(),
      epoch.Type.TEST: set(),
    }

    # Register this model with the logger.
    self.logger.OnStartRun(self.run_id, self.graph_db)

  #############################################################################
  # Interface methods. Subclasses must implement these.
  #############################################################################

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch of data from an iterator of graphs.

    Implementations of this method must be thread safe. Multiple threads may
    concurrently call this method using different graph iterators. This is to
    amortize I/O costs when alternating between training / validation / testing
    datasets.

    Returns:
      A single batch of data for feeding into RunBatch(). A batch consists of a
      list of graph IDs and a model-defined blob of data. If the list of graph
      IDs is empty, the batch is discarded and not fed into RunBatch().
    """
    raise NotImplementedError("abstract class")

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batches.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Results:
    """Process a mini-batch of data using the model.

    Args:
      log: The mini-batch log returned by MakeBatch().
      batch: The batch data returned by MakeBatch().

    Returns:
      The target values for the batch, and the predicted values.
    """
    raise NotImplementedError("abstract class")

  def CreateModelData(self) -> None:
    """Initialize the starting state of a model.

    Use this method to perform any model-specific initialisation such as
    randomizing starting weights. When restoring a model from a checkpoint, this
    method is *not* called. Instead, LoadModelData() will be called.

    Note that subclasses must call this superclass method first.
    """
    pass

  def LoadModelData(self, data_to_load: Any) -> None:
    """Set the model state from the given model data.

    Args:
      data_to_load: The return value of GetModelData().
    """
    raise NotImplementedError("abstract class")

  def GetModelData(self) -> Any:
    """Return the model state.

    Returns:
      A  model-defined blob of data that can later be passed to LoadModelData()
      to restore the current model state.
    """
    raise NotImplementedError("abstract class")

  def Summary(self) -> str:
    """Return a long summary string describing the model."""
    return type(self).__name__

  def NeedsGraphTuples(self) -> bool:
    return True

  #############################################################################
  # Automatic methods.
  #############################################################################

  def __call__(
    self,
    epoch_type: epoch.Type,
    batch_iterator: batches.BatchIterator,
    logger: logging.Logger,
  ) -> epoch.Results:
    """Run the model for over the input batches.

    This is the heart of the model - where you run an epoch of batches through
    the graph and produce results. The interface for training and inference is
    the same, only the epoch_type value should change.

    Side effects of calling a model are:
      * The model bumps its epoch_num counter if on a training epoch.
      * The model updates its best_results dictionary if the accuracy produced
        by this epoch is greater than the previous best.

    Args:
      epoch_type: The type of epoch to run.
      batch_iterator: The batches to process.
      logger: A logger instance to log results to.

    Returns:
      An epoch results instance.
    """
    if not self._initialized:
      raise TypeError(
        "Model called before Initialize() or FromCheckpoint() invoked"
      )

    # Only training epochs bumps the epoch count.
    if epoch_type == epoch.Type.TRAIN:
      self.epoch_num += 1

    thread = EpochThread(self, epoch_type, batch_iterator, logger)
    progress.Run(thread)

    # Check that there were batches.
    if not thread.batch_count:
      raise ValueError("No batches")

    # If --strict_graph_segmentation is set, check for graphs that we have
    # already seen before.
    if FLAGS.strict_graph_segmentation:
      with logger.ctx.Profile(4, "Checked strict graph segmentation"):
        for other_epoch_type in set(list(epoch.Type)) - {epoch_type}:
          duplicate_graph_ids = self.graph_ids[other_epoch_type].intersection(
            thread.graph_ids
          )
          if duplicate_graph_ids:
            raise ValueError(
              f"{epoch_type} batch contains {len(duplicate_graph_ids)} graphs "
              f"from {other_epoch_type}: {list(duplicate_graph_ids)[:100]}"
            )
        self.graph_ids[epoch_type] = self.graph_ids[epoch_type].union(
          thread.graph_ids
        )

    # TODO(github.com/ChrisCummins/ProGraML/issues/38): Explicitly free the
    # thread object to see if that is contributing to climbing memory usage.
    results = copy.deepcopy(thread.results)
    if not results:
      raise OSError("Epoch produced no results. Did the model crash?")
    del thread

    # Update the record of best results.
    if results > self.best_results[epoch_type].results:
      new_best = epoch.BestResults(epoch_num=self.epoch_num, results=results)
      logger.ctx.Log(
        2,
        "%s results improved from %s",
        epoch_type.name.capitalize(),
        self.best_results[epoch_type],
      )
      self.best_results[epoch_type] = new_best

    return results

  def BatchIterator(
    self,
    epoch_type: epoch.Type,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> Iterable[batches.Data]:
    """Generate model batches from a iterator of graphs.

    Args:
      epoch_type: The type of epoch that batches are being constructed for.
        This is used only for logging printouts.
      graphs: The graphs to construct batches from.
      ctx: A logging context.

    Returns:
      A batch iterator.
    """
    while True:
      with ctx.Profile(
        4,
        lambda t: (
          f"Constructed batch of "
          f"{humanize.Plural(batch.graph_count, f'{epoch_type.name.lower()} graph')}"
        ),
      ):
        batch = self.MakeBatch(graphs)

      # A batch with no graphs is considered "empty" and is not returned.
      if batch.graph_count:
        yield batch
      else:
        break

  def Initialize(self) -> None:
    """Initialize an untrained model."""
    if self._initialized:
      raise TypeError("CreateModelData() called on already-initialized model")

    self._initialized = True
    self.CreateModelData()

  def RestoreFrom(self, checkpoint_ref: checkpoints.CheckpointReference):
    """Restore a model from a checkpoint."""
    self._initialized = True
    self.restored_from = checkpoint_ref
    checkpoint = self.logger.Load(checkpoint_ref)
    self.epoch_num = checkpoint.epoch_num
    self.best_results = checkpoint.best_results
    self.LoadModelData(checkpoint.model_data)

  def SaveCheckpoint(self) -> checkpoints.CheckpointReference:
    """Construct a checkpoint from the current model state.

    Returns:
      A checkpoint reference.
    """
    if not self._initialized:
      raise TypeError("Cannot save an unitialized model.")

    self.logger.Save(
      checkpoints.Checkpoint(
        run_id=self.run_id,
        epoch_num=self.epoch_num,
        best_results=self.best_results,
        model_data=self.GetModelData(),
      )
    )
    return checkpoints.CheckpointReference(
      run_id=self.run_id, epoch_num=self.epoch_num
    )

  @decorators.memoized_property
  def parameters(self) -> pd.DataFrame:
    return self.logger.GetParameters(self.run_id)


class EpochThread(progress.Progress):
  """A thread which runs a single epoch of a model.

  After running this thread, the results of the epoch may be accessed through
  the 'results' parameter.
  """

  def __init__(
    self,
    model: ClassifierBase,
    epoch_type: epoch.Type,
    batch_iterator: batches.BatchIterator,
    logger: logging.Logger,
  ):
    """Constructor.

    Args:
      model: A model instance.
      epoch_type: The type of epoch to run.
      batch_iterator: A batch iterator.
      logger: A logger.
    """
    self.model = model
    self.epoch_type = epoch_type
    self.batch_iterator = batch_iterator
    self.logger = logger
    self.batch_count = 0

    # Set at the end of Run().
    self.results: epoch.Results = None
    self.graph_ids = set()

    super(EpochThread, self).__init__(
      f"{epoch_type.name.capitalize()} epoch {model.epoch_num}",
      0,
      batch_iterator.graph_count,
      unit="graph",
      vertical_position=0,
      leave=False,
    )

  def Run(self) -> None:
    """Run the epoch worker thread."""
    rolling_results = batches.RollingResults()

    for i, batch in enumerate(self.batch_iterator.batches):
      self.batch_count += 1
      self.ctx.i += batch.graph_count

      # Record the graph IDs.
      for graph_id in batch.graph_ids:
        self.graph_ids.add(graph_id)

      # Check that at least one batch is produced.
      if not i and not batch.graph_count:
        raise OSError("No batches generated!")

      # We have run out of graphs.
      if not batch.graph_count:
        break

      # Run the batch through the model.
      with self.ctx.Profile(
        3,
        lambda t: (
          f"Batch {i+1} with "
          f"{batch.graph_count} graphs: "
          f"{batch_results}"
        ),
      ) as batch_timer:
        batch_results = self.model.RunBatch(self.epoch_type, batch)

      # Record the batch results.
      self.logger.OnBatchEnd(
        run_id=self.model.run_id,
        epoch_type=self.epoch_type,
        epoch_num=self.model.epoch_num,
        batch_num=i + 1,
        timer=batch_timer,
        data=batch,
        results=batch_results,
      )
      rolling_results.Update(
        batch, batch_results, weight=batch_results.target_count
      )
      self.ctx.bar.set_postfix(
        loss=rolling_results.loss,
        acc=rolling_results.accuracy,
        prec=rolling_results.precision,
        rec=rolling_results.recall,
      )

    self.results = epoch.Results.FromRollingResults(rolling_results)
    self.logger.OnEpochEnd(
      self.model.run_id, self.epoch_type, self.model.epoch_num, self.results
    )
