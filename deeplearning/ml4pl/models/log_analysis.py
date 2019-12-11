"""A module for analyzing log databases."""
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.metrics

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import decorators
from labm8.py import progress

FLAGS = app.FLAGS


class RunLogAnalyzer(object):
  """Analyse the logs of a single run."""

  def __init__(
    self,
    log_db: log_database.Database,
    run_id: run_id_lib.RunId,
    graph_db: Optional[graph_tuple_database.Database] = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    self.log_db = log_db
    self.run_id = run_id
    self.ctx = ctx
    self._graph_db = graph_db

    # Check that the requested run exists in the database.
    with self.log_db.Session() as session:
      if (
        not session.query(log_database.RunId)
        .filter(log_database.RunId.run_id == str(run_id))
        .scalar()
      ):
        raise ValueError(f"Run not found: {self.run_id}")

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
        log_database.Parameter.run_id == str(self.run_id),
        log_database.Parameter.name == "graph_db",
      ).scalar()
      if not graph_param:
        raise ValueError("Unable to determine graph_db flag")
      graph_db_url = str(graph_param.value)

    return graph_tuple_database.Database(graph_db_url, must_exist=True)

  @decorators.memoized_property
  def tables(self) -> Dict[str, pd.DataFrame]:
    """Get the {parameters, epochs, runs} tables for the run."""
    return {
      name: df for name, df in self.log_db.GetTables(run_ids=[self.run_id])
    }

  @decorators.memoized_property
  def best_results(self) -> Dict[epoch.Type, epoch.BestResults]:
    """Get the best results dict.

    Returns:
      A mapping from <epoch_type, epoch.BestResults> for the best accuracy on
      each of the epoch types.
    """
    return self.log_db.GetBestResults(run_id=self.run_id)

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
        raise ValueError(f"No {self.run_id} epochs reached {metric}")
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
        output_graph = input_graph.SetLabels(
          node_y=predictions[node_count : node_count + input_graph.node_count],
          copy=False,
        )
        node_count += input_graph.node_count
        yield input_graph, output_graph
    elif input_graphs[0].graph_y_dimensionality:
      # Graph-level predictions:
      for graph_count, graph in enumerate(input_graphs):
        input_graph = graph.tuple
        output_graph = input_graph.SetLabels(
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
