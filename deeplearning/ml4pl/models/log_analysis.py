"""A module for analyzing log databases."""
import pathlib
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.metrics
import sqlalchemy as sql
from matplotlib import pyplot as plt
from matplotlib import ticker

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import export_logs
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import decorators
from labm8.py import progress
from labm8.py import sqlutil

# export_logs module is required to pull in dependencies required to construct
# summary tables.
del export_logs

FLAGS = app.FLAGS

app.DEFINE_output_path(
  "log_analysis_outdir",
  None,
  "When //deeplearning/ml4pl/models:log_analysis is executed as a script, this "
  "determines the directory to write files to.",
)


class LogAnalyzer(object):
  """Analyze the logs in a database."""

  def __init__(
    self,
    log_db: log_database.Database,
    run_ids: List[run_id_lib.RunId] = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    self.log_db = log_db
    self.run_ids = run_ids
    self.ctx = ctx

    # Check that the requested run exists in the database.
    if not self.log_db.run_ids:
      raise ValueError("Log database is empty")
    for run_id in self.run_ids:
      if str(run_id) not in self.log_db.run_ids:
        raise ValueError(f"Run ID not found: {run_id}")

  @decorators.memoized_property
  def tables(self) -> Dict[str, pd.DataFrame]:
    """Get the {parameters, epochs, runs} tables for the run."""
    return {
      name: df for name, df in self.log_db.GetTables(run_ids=self.run_ids)
    }

  def PlotEpochMetrics(
    self, metric: str, epoch_types: List[str] = None, ax=None,
  ) -> None:
    """Plot a metric over epochs.

    Args:
      metric: The metric of interest.
      epoch_types: The epoch types to plot. A list of {train,val,test} values.
      ax: An axis to plot on.
    """
    # Set default argument values.
    epoch_typenames = epoch_types or ["train", "val", "test"]
    ax = ax or plt.gca()

    # Read the epochs table.
    df = self.tables["epochs"]

    # Check that the requested metric exists.
    if f"train_{metric}" not in df:
      available_metrics = sorted(
        c[len("train_") :] for c in df.columns.values if c.startswith("train_")
      )
      raise ValueError(
        f"No such metric: {metric}. Available metrics: {available_metrics}"
      )

    # Create the metric plot for each run.
    for run_id in set(df["run_id"].values):
      for epoch_type in epoch_typenames:
        metric_name = f"{epoch_type}_{metric}"
        run_df = df[(df["run_id"] == run_id) & (df[metric_name].notnull())]
        if len(run_df):
          x = run_df["epoch_num"].values
          y = run_df[metric_name].values
          ax.plot(x, y, label=f"{run_id}:{epoch_type}")

    # Configure the Y axis.
    ax.set_ylabel(metric.capitalize())

    # Configure the X axis.
    ax.set_xlabel("Epoch")
    # Force integer axis for epoch number.
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Force the legend.
    plt.legend()


class RunLogAnalyzer(LogAnalyzer):
  """Analyse the logs of a single run."""

  def __init__(
    self,
    log_db: log_database.Database,
    run_id: run_id_lib.RunId,
    graph_db: Optional[graph_tuple_database.Database] = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    super(RunLogAnalyzer, self).__init__(
      log_db=log_db, run_ids=[run_id], ctx=ctx
    )
    self._graph_db = graph_db

  @decorators.memoized_property
  def graph_db(self) -> graph_tuple_database.Database:
    """Return the graph database for a run. This is reconstructed from the
    --graph_db flag value recorded for the run."""
    if self._graph_db:
      return self._graph_db

    with self.log_db.Session() as session:
      graph_param: log_database.Parameter = session.query(
        log_database.Parameter
      ).filter(
        log_database.Parameter.type_num
        == log_database.ParameterType.FLAG.value,
        log_database.Parameter.run_id == str(self.run_ids[0]),
        log_database.Parameter.name == "graph_db",
      ).scalar()
      if not graph_param:
        raise ValueError("Unable to determine graph_db flag")
      graph_db_url = str(graph_param.value)

    return graph_tuple_database.Database(graph_db_url, must_exist=True)

  @decorators.memoized_property
  def best_results(self) -> Dict[epoch.Type, epoch.BestResults]:
    """Get the best results dict.

    Returns:
      A mapping from <epoch_type, epoch.BestResults> for the best accuracy on
      each of the epoch types.
    """
    return self.log_db.GetBestResults(run_id=self.run_ids[0])

  def GetBestEpochNum(self, metric="best accuracy") -> int:
    """Select the train/val/test epoch logs using the given metric.

    Supported metrics are:
      best accuracy
      best precision
      best recall
      best f1
      90% val acc
      95% val acc
      99% val acc
      99.9% val acc
    """
    epochs: pd.DataFrame = self.tables["epochs"]

    if metric in {"best accuracy", "best precision", "best recall", "best f1"}:
      column_name = "val_" + metric[len("best ") :]
      best_epoch = epochs.iloc[epochs[column_name].idxmax()]
      epoch_num = best_epoch.epoch_num
    elif metric in {
      "90% val acc",
      "95% val acc",
      "99% val acc",
      "99.9% val acc",
    }:
      accuracy = float(metric.split("%")[0]) / 100
      matching_rows = epochs[epochs["val_accuracy"] >= accuracy]
      if not len(matching_rows):
        raise ValueError(f"No {self.run_ids[0]} epochs reached {metric}")
      # Select the first epoch when there are more than one matching rows.
      epoch_num = epochs.iloc[matching_rows.index[0]].epoch
    else:
      raise ValueError(f"Unknown metric `{metric}`")

    return epoch_num

  def GetGraphsForBatch(
    self, batch: log_database.Batch
  ) -> Iterable[graph_tuple_database.GraphTuple]:
    """Reconstruct the graphs for a batch.

    Returns:
      A iterable sequence of the unique graphs from a batch. Note that order may
      not be the same as the order they appeared in the batch, and that
      duplicate graphs in the batch will only be returned once.
    """
    if not batch.details:
      raise OSError("Cannot re-create batch without detailed logs")

    filters = [lambda: graph_tuple_database.GraphTuple.id.in_(batch.graph_ids)]
    return graph_database_reader.BufferedGraphReader(
      self.graph_db, filters=filters
    )

  def GetInputOutputGraphs(
    self, batch: log_database.Batch
  ) -> Iterable[Tuple[graph_tuple.GraphTuple, graph_tuple.GraphTuple]]:
    """Reconstruct the input/output graphs for a batch.

    This returns the raw GraphTuples for a batch, with node_y or graph_y
    attributes on the output graphs set to the raw model predictions.
    """
    # Read the graphs from the batch.
    unique_graphs: List[graph_tuple_database.GraphTuple] = list(
      self.GetGraphsForBatch(batch)
    )
    id_to_graphs = {graph.id: graph for graph in unique_graphs}

    # Re-construct the full set of graphs from the batch.
    input_graphs: List[graph_tuple_database.GraphTuple] = [
      id_to_graphs[id] for id in batch.graph_ids
    ]

    # Reconstruct the output graphs.
    predictions = batch.predictions

    if input_graphs[0].node_y_dimensionality:
      # Node-level predictions:
      node_count = 0
      for graph in input_graphs:
        input_graph = graph.tuple
        output_graph = input_graph.SetFeaturesAndLabels(
          node_y=predictions[node_count : node_count + input_graph.node_count],
          copy=False,
        )
        node_count += input_graph.node_count
        yield input_graph, output_graph
    elif input_graphs[0].graph_y_dimensionality:
      # Graph-level predictions:
      for graph_count, graph in enumerate(input_graphs):
        input_graph = graph.tuple
        output_graph = input_graph.SetFeaturesAndLabels(
          graph_y=predictions[graph_count], copy=False
        )
        yield input_graph, output_graph
    else:
      raise NotImplementedError("neither node_y or graph_y set")

  @staticmethod
  def NodeConfusionMatrix(
    input_graph: graph_tuple.GraphTuple, output_graph: graph_tuple.GraphTuple
  ) -> pd.DataFrame:
    """Build a confusion matrix for the given input/output graph pair."""
    targets = input_graph.node_y
    predictions = output_graph.node_y

    confusion_matrix = BuildConfusionMatrix(
      targets=targets, predictions=predictions
    )
    return pd.DataFrame(
      confusion_matrix,
      columns=[f"pred_{i}" for i in range(len(confusion_matrix))],
      index=[f"true_{i}" for i in range(len(confusion_matrix))],
    )


def SortGraphsByAccuracy(
  input_output_graphs: Iterable[
    Tuple[graph_tuple.GraphTuple, graph_tuple.GraphTuple]
  ]
) -> List[Tuple[graph_tuple.GraphTuple, graph_tuple.GraphTuple]]:
  """Sort the list of input/output graphs by their accuracy."""
  input_output_graphs = list(input_output_graphs)
  return sorted(input_output_graphs, key=lambda x: ComputeGraphAccuracy(*x))


def ComputeGraphAccuracy(
  input_graph: graph_tuple.GraphTuple, output_graph: graph_tuple.GraphTuple,
):
  """Return the classification accuracy of the given input/output graph.

  Supports node-level or graph-level labels.

  Returns:
    Accuracy in the range 0 <= x <= 1.
  """
  if input_graph.has_node_y:
    true_y = np.argmax(input_graph.node_y, axis=1)
    pred_y = np.argmax(output_graph.node_y, axis=1)
  elif input_graph.has_graph_y:
    true_y = np.argmax(input_graph.graph_y)
    pred_y = np.argmax(output_graph.graph_y)
  else:
    raise NotImplementedError("unreachable")

  return sklearn.metrics.accuracy_score(true_y, pred_y)


def BuildConfusionMatrix(targets: np.array, predictions: np.array) -> np.array:
  """Build a confusion matrix.

  Args:
    targets: A list of 1-hot vectors with shape
      (num_instances, num_classes), dtype int32.
    predictions: A list of 1-hot vectors with shape
      (num_instances, num_classes), dtype float32.

  Returns:
    A pickled confusion matrix, which is a matrix of shape
    [num_classes, num_classes] where the rows indicate true target class,
    the columns indicate predicted target class, and the element values are
    the number of instances of this type in the batch.
  """
  if targets.shape != predictions.shape:
    raise TypeError(
      f"Predictions shape {predictions.shape} must match targets "
      f"shape {targets.shape}"
    )

  num_classes = targets.shape[1]

  # Convert 1-hot vectors to dense lists of integers.
  targets = np.argmax(targets, axis=1)
  predictions = np.argmax(predictions, axis=1)

  confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
  for target, prediction in zip(targets, predictions):
    confusion_matrix[target][prediction] += 1

  return confusion_matrix


class WriteGraphsToFile(progress.Progress):
  """Write graphs in a graph database to pickled files.

  This is for debugging.
  """

  def __init__(self, outdir: pathlib.Path):
    self.outdir = outdir
    self.outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_ref = checkpoints.CheckpointReference.FromString(FLAGS.run_id)

    self.analysis = RunLogAnalyzer(
      log_db=FLAGS.log_db(), run_id=checkpoint_ref.run_id,
    )
    self.epoch_num = 0

    with self.analysis.log_db.Session() as session:
      detailed_batch_graph_count = (
        session.query(sql.func.sum(log_database.Batch.graph_count))
        .filter(
          log_database.Batch.run_id == str(checkpoint_ref.run_id),
          log_database.Batch.epoch_num == self.epoch_num,
        )
        .join(log_database.BatchDetails)
        .scalar()
      )

    super(WriteGraphsToFile, self).__init__(
      "analyzer", i=0, n=detailed_batch_graph_count
    )
    self.analysis.ctx = self.ctx

  def Run(self):
    """Read and write the graphs."""
    # GetInputOutputGraphs
    with self.analysis.log_db.Session() as session:
      query = (
        session.query(log_database.Batch)
        .options(sql.orm.joinedload(log_database.Batch.details))
        .filter(
          log_database.Batch.run_id == str(self.analysis.run_id),
          log_database.Batch.epoch_num == self.epoch_num,
        )
        .join(log_database.BatchDetails)
      )

      for i, batches in enumerate(
        sqlutil.OffsetLimitBatchedQuery(query, batch_size=512)
      ):
        for batch in batches.rows:
          self.ctx.i += batch.graph_count
          for input_graph, output_graph in self.analysis.GetInputOutputGraphs(
            batch
          ):
            input_path = self.outdir / f"graph_tuple_{i:08}_input.pickle"
            output_path = self.outdir / f"graph_tuple_{i:08}_output.pickle"
            input_graph.ToFile(input_path)
            output_graph.ToFile(output_path)


def Main():
  """Main entry point."""
  if not FLAGS.log_analysis_outdir:
    raise app.UsageError("--log_analysis_outdir must be set")
  progress.Run(WriteGraphsToFile(FLAGS.log_analysis_outdir))


if __name__ == "__main__":
  app.Run(Main)
