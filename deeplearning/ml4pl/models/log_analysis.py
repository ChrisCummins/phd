"""A module for analyzing log databases."""
import random
import typing

import networkx as nx
import numpy as np
import pandas as pd
import sqlalchemy as sql
from labm8 import app
from labm8 import decorators
from labm8 import humanize
from labm8 import labtypes
from labm8 import prof

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.graph_dict import graph_batcher
from deeplearning.ml4pl.models import log_database

FLAGS = app.FLAGS


class RunLogAnalyzer(object):
  """Analyse the logs of a single run."""

  def __init__(self, graph_db: graph_database.Database,
               log_db: log_database.Database, run_id: str):
    self.graph_db = graph_db
    self.log_db = log_db
    self.run_id = run_id

    self.batcher = graph_batcher.GraphBatcher(
        self.graph_db,
        # We can pass any non-zero value for message_passing_step_count, it
        # won't be used.
        message_passing_step_count=1)

    with self.log_db.Session() as session:
      num_logs = session.query(log_database.BatchLog.run_id) \
        .filter(log_database.BatchLog.run_id == self.run_id) \
        .count()
      if not num_logs:
        raise ValueError(f"Run `{self.run_id}` not found in log database")

    app.Log(1, "Found %s logs for run `%s`", humanize.Commas(num_logs),
            self.run_id)

  @decorators.memoized_property
  def batch_logs(self) -> pd.DataFrame:
    with prof.Profile(f'Read batch logs'):
      return self.log_db.BatchLogsToDataFrame(self.run_id, per_global_step=True)

  @decorators.memoized_property
  def epoch_logs(self) -> pd.DataFrame:
    with prof.Profile(f'Read epoch logs'):
      return self.log_db.BatchLogsToDataFrame(self.run_id,
                                              per_global_step=False)

  def GetEpochLogs(self, epoch_num: int) -> pd.DataFrame:
    """Return the logs for the given epoch number, index by group."""
    if epoch_num not in self.epoch_logs.epoch:
      raise ValueError(f"Epoch `{epoch_num}` not in logs: "
                       f"{set(self.epoch_logs.epoch)}")
    return self.epoch_logs[self.epoch_logs.epoch == epoch_num].set_index(
        'group')

  def GetBestEpoch(self, metric='best accuracy'):
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
    if metric in {'best accuracy', 'brest precision', 'best recall', 'best f1'}:
      column = metric[len('best '):]
      validation_metric = getattr(
          self.epoch_logs[self.epoch_logs['group'] == 'val'], column)
      best_validation_metric = self.epoch_logs.iloc[validation_metric.idxmax()]
      epoch_num = best_validation_metric.epoch
    elif metric in {
        '90% val acc', '95% val acc', '99% val acc', '99.9% val acc'
    }:
      accuracy = float(metric.split('%')[0])
      matching_rows = self.epoch_logs[(self.epoch_logs['group'] == 'val') &
                                      (self.epoch_logs['accuracy'] > accuracy)]
      epoch_num = self.epoch_logs.iloc[matching_rows.index[0]].epoch
    else:
      raise ValueError(f"Unknown metric `{metric}`")

    return self.GetEpochLogs(epoch_num)

  def ReconstructBatchDict(self, log: log_database.BatchLog):
    """Reconstruct the batch dict for the given log."""
    with prof.Profile(lambda t: ('Reconstructed batch dict with '
                                 f'{batch_dict["graph_count"]} graphs')):
      with self.graph_db.Session() as session:
        graphs = session.query(graph_database.GraphMeta) \
          .options(sql.orm.joinedload(graph_database.GraphMeta.graph)) \
          .filter(graph_database.GraphMeta.id.in_(log.graph_indices)).all()
        # Load the graphs in the same order as in the batch.
        graphs = sorted(graphs, key=lambda g: log.graph_indices.index(g.id))

      batch_dict = self.batcher.CreateBatchDict((g for g in graphs))

    return batch_dict

  def GetInputOutputGraphsForRandomBatch(self,
                                         epoch_num: int,
                                         group: str = 'val'):
    """Fetch the batch dict for a random batch from the given epoch_num where
    the accuracy was < 100%.
    """
    batches = self.batch_logs[(self.batch_logs['epoch'] == epoch_num) &
                              (self.batch_logs['group'] == group) &
                              (self.batch_logs['accuracy'] < 100)]

    random_row = batches.iloc[random.randint(0, len(batches) - 1)]

    with self.log_db.Session() as session:
      log = session.query(log_database.BatchLog) \
        .filter(log_database.BatchLog.run_id == self.run_id) \
        .filter(log_database.BatchLog.global_step == random_row.global_step) \
        .one()

    batch_dict = self.ReconstructBatchDict(log)

    with prof.Profile(f"Recreated {batch_dict['graph_count']} input graphs"):
      input_graphs = list(self.batcher.BatchDictToGraphs(batch_dict))

    with prof.Profile(f"Recreated {batch_dict['graph_count']} output graphs"):
      # Remove the features.
      labtypes.DeleteKeys(batch_dict, {'node_x', 'edge_x', 'graph_x'})

      # Add the labels.
      if 'node_y' in batch_dict:
        batch_dict['node_y'] = log.predictions
      elif 'graph_y' in batch_dict:
        batch_dict['graph_y'] = log.predictions
      else:
        raise ValueError(
            "Neither node or graph labels found in batch dict with "
            f"keys: `{list(batch_dict.keys())}`")

      output_graphs = list(self.batcher.BatchDictToGraphs(batch_dict))

    return list(zip(input_graphs, output_graphs))

  @staticmethod
  def NodeConfusionMatrix(input_graph: nx.MultiDiGraph,
                          output_graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Build a confusion matrix for the given input/output graph pair."""
    targets = np.array([data['y'] for _, data in input_graph.nodes(data=True)])
    predictions = np.array(
        [data['y'] for _, data in output_graph.nodes(data=True)])

    cm = BuildConfusionMatrix(targets=targets, predictions=predictions)
    return pd.DataFrame(cm,
                        columns=[f'pred_{i}' for i in range(len(cm))],
                        index=[f'true_{i}' for i in range(len(cm))])


def SortGraphsByAccuracy(
    input_output_graphs: typing.List[
        typing.Tuple[nx.MultiDiGraph, nx.MultiDiGraph]]
) -> typing.List[typing.Tuple[nx.MultiDiGraph, nx.MultiDiGraph]]:
  """Sort the list of input/output graphs by their accuracy."""
  return sorted(input_output_graphs, key=lambda x: ComputeGraphAccuracy(*x))


def ComputeGraphAccuracy(input_graph: nx.MultiDiGraph,
                         output_graph: nx.MultiDiGraph):
  """Return the classification accuracy of the given input/output graph.

  Supports node-level or graph-level labels.

  Returns:
    Accuracy in the range 0 <= x <= 1.
  """
  try:
    accuracy = int(input_graph.y == output_graph.y)
  except AttributeError:
    true_y = np.argmax([y for _, y in input_graph.nodes(data='y')], axis=1)
    pred_y = np.argmax([y for _, y in output_graph.nodes(data='y')], axis=1)
    correct = (true_y == pred_y)
    for i, (_, data) in enumerate(output_graph.nodes(data=True)):
      data['correct'] = correct[i]
    accuracy = (true_y == pred_y).mean()
  output_graph.accuracy = accuracy
  return accuracy


def BuildConfusionMatrix(targets: np.array, predictions: np.array) -> np.array:
  """Build a confusion matrix.

  Args:
    targets: A list of 1-hot vectors with shape [num_instances,num_classes].
    predictions: A list of 1-hot vectors with shape [num_instances,num_classes].

  Returns:
    A pickled confusion matrix, which is a matrix of shape
    [num_classes, num_classes] where the rows indicate true target class,
    the columns indicate predicted target class, and the element values are
    the number of instances of this type in the batch.
  """
  num_classes = len(targets[0])

  # Convert 1-hot vectors to dense lists of integers.
  targets = np.argmax(targets, axis=1)
  predictions = np.argmax(predictions, axis=1)

  confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
  for target, prediction in zip(targets, predictions):
    confusion_matrix[target][prediction] += 1

  return confusion_matrix
