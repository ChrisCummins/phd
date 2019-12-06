"""Base class for implementing classifier models."""
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batchs
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import logger as logging
from labm8.py import app
from labm8.py import progress


FLAGS = app.FLAGS


class ClassifierBase(object):
  """Abstract base class for implementing classifiers.

  Before feeding any data through a model, you must call Initialize(). Else
  use the FromCheckpoint() class constructor to construct and initialize a
  model from a checkpoint.

  Subclasses must implement the following methods:
    MakeBatch()        # construct a batch from input graphs.
    RunBatch()         # run the model on the batch.

  And may optionally wish to implement these additional methods:
    CreateModelData()  # initialize an untrained model.
    GetModelData()     # get model data to save.
    LoadModelData()    # load model data.
  """

  def __init__(
    self,
    node_y_dimensionality: int,
    graph_y_dimensionality: int,
    edge_position_max: int,
    restored_from: Optional[run_id_lib.RunId] = None,
    run_id: Optional[run_id_lib.RunId] = None,
  ):
    """Constructor.

    Args:
      node_y_dimensionality: The dimensionality of per-node labels.
      graph_y_dimensionality: The dimensionality of per-graph labels.
      edge_position_max: The maximum edge position.

    Raises:
      NotImplementedError: If both node and graph labels are set.
      TypeError: If neither graph or node labels are set.
    """
    # Set by FromCheckpoint() or Initialize().
    self._initialized = False

    # The unique ID of this model instance.
    self.run_id: run_id_lib.RunId = (
      run_id or run_id_lib.RunId.GenerateUnique(type(self).__name__)
    )

    # Model properties.
    self.restored_from = restored_from
    self.node_y_dimensionality = node_y_dimensionality
    self.graph_y_dimensionality = graph_y_dimensionality
    self.edge_position_max = edge_position_max

    # Determine the label dimensionality.
    if node_y_dimensionality and graph_y_dimensionality:
      raise NotImplementedError(
        "Both node and graph labels are set. This is currently not supported. "
        "See <github.com/ChrisCummins/ProGraML/issues/26>"
      )
    self.y_dimensionality = node_y_dimensionality or graph_y_dimensionality
    if not self.y_dimensionality:
      raise TypeError("Neither node or graph dimensionalities are set")

    # Progress counters. These are saved and restored from file.
    self.epoch_num = 0
    self.best_results: Dict[epoch.Type, epoch.BestResults] = {
      epoch.Type.TRAIN: epoch.BestResults(),
      epoch.Type.VAL: epoch.BestResults(),
      epoch.Type.TEST: epoch.BestResults(),
    }

  #############################################################################
  # Interface methods. Subclasses must implement these.
  #############################################################################

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batchs.Data:
    """Create a mini-batch of data from an iterator of graphs.

    Implementatinos of this method must be thread safe. Multiple threads may
    concurrently call this method using different graph iterators. This is to
    amortize I/O costs when alternating between training / validation / testing
    datasets.

    Returns:
      A single batch of data for feeding into RunBatch().
    """
    raise NotImplementedError("abstract class")

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batchs.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batchs.Results:
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
    return None

  def GetModelData(self) -> Any:
    """Return the model state.

    Returns:
      A  model-defined blob of data that can later be passed to LoadModelData()
      to restore the current model state.
    """
    return None

  #############################################################################
  # Automatic methods.
  #############################################################################

  def BatchIterator(
    self, graphs: Iterable[graph_tuple_database.GraphTuple]
  ) -> Iterable[batchs.Data]:
    """Generate model batches from an input graph iterator.

    Args:
      graphs: The graphs to construct batches from.

    Returns:
      A batch iterator.
    """
    while True:
      batch = self.MakeBatch(graphs)
      if batch.graph_count:
        yield batch
      else:
        break

  def __call__(
    self,
    epoch_type: epoch.Type,
    batch_iterator: batchs.BatchIterator,
    logger: logging.Logger,
  ) -> epoch.Results:
    """Run the model over the inputs.

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

    thread = EpochThread(self, epoch_type, batch_iterator, logger)
    progress.Run(thread)
    return thread.results

  def UpdateBestResults(
    self,
    epoch_type: epoch.Type,
    epoch_num: int,
    results: epoch.Results,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> bool:
    """Update the 'best results' state of a model.

    Args:
      epoch_type: The epoch type of the new results.
      epoch_num: The epoch number of the new results.
      results: The new epoch results.
      ctx: A logging context.

    Returns:
      True if the new results are an improvement over the previous best, else
      False.
    """
    if results > self.best_results[epoch_type].results:
      new_best = epoch.BestResults(epoch_num=epoch_num, results=results)
      ctx.Log(
        2,
        "%s results improved:\n  from: %s\n    to: %s",
        epoch_type.name.capitalize(),
        self.best_results[epoch_type],
        new_best,
      )
      self.best_results[epoch_type] = new_best
      return True
    else:
      return False

  #############################################################################
  # Initializing, restoring, and saving models.
  #############################################################################

  def Initialize(self) -> None:
    """Initialize an untrained model."""
    if self._initialized:
      raise TypeError("CreateModelData() called on already-initialized model")

    self._initialized = True
    self.CreateModelData()

  @classmethod
  def FromCheckpoint(
    cls,
    checkpoint: checkpoints.Checkpoint,
    node_y_dimensionality: int,
    graph_y_dimensionality: int,
    edge_position_max: int,
  ):
    """Construct a model from checkpoint data."""
    model = cls(
      node_y_dimensionality=node_y_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      edge_position_max=edge_position_max,
      restored_from=checkpoint.run_id,
    )
    model._initialized = True
    model.epoch_num = checkpoint.epoch_num
    model.best_results = checkpoint.best_results
    model.LoadModelData(checkpoint.model_data)
    return model

  def GetCheckpoint(self) -> checkpoints.Checkpoint:
    """Construct a checkpoint from the current model state.

    Returns:
      A checkpoint instance.
    """
    return checkpoints.Checkpoint(
      run_id=self.run_id,
      epoch_num=self.epoch_num,
      best_results=self.best_results,
      model_data=self.GetModelData(),
    )


class EpochThread(progress.Progress):
  """A thread which runs a single epoch of a model.

  After running this thread, the results of the epoch may be accessed through
  the 'results' parameter.
  """

  def __init__(
    self,
    model: ClassifierBase,
    epoch_type: epoch.Type,
    batch_iterator: batchs.BatchIterator,
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

    # Set at the end of Run().
    self.results: epoch.Results = epoch.Results()

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
    rolling_results = batchs.RollingResults()

    for i, batch in enumerate(self.batch_iterator.batches):
      self.ctx.i += batch.graph_count

      # Check that at least one batch is produced.
      if not i and not batch.graph_count:
        raise OSError("No batches generated!")

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
      rolling_results.Update(batch_results)
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
