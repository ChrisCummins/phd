"""A module for analyzing log databases."""
import random
import typing

import networkx as nx
import numpy as np
import pandas as pd
import sklearn.metrics
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
from deeplearning.ml4pl.models import log_database
from labm8 import app
from labm8 import decorators
from labm8 import humanize
from labm8 import prof

FLAGS = app.FLAGS


class RunLogAnalyzer(object):
  """Analyse the logs of a single run."""

  def __init__(self, graph_db: graph_database.Database,
               log_db: log_database.Database, run_id: str):
    self.graph_db = graph_db
    self.log_db = log_db
    self.run_id = run_id

    # A graph batcher is used to re-construct networkx graphs from the graph
    # tuples in the database.
    self.batcher = graph_batcher.GraphBatcher(self.graph_db)

    with self.log_db.Session() as session:
      num_logs = session.query(log_database.BatchLogMeta.run_id) \
        .filter(log_database.BatchLogMeta.run_id == self.run_id) \
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
    """Return the logs for the given epoch number, index by type."""
    if epoch_num not in self.epoch_logs.epoch.values:
      raise ValueError(f"Epoch `{epoch_num}` not in logs: "
                       f"{set(self.epoch_logs.epoch)}")
    return self.epoch_logs[self.epoch_logs.epoch == epoch_num].set_index('type')

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
          self.epoch_logs[self.epoch_logs['type'] == 'val'], column)
      best_validation_metric = self.epoch_logs.iloc[validation_metric.idxmax()]
      epoch_num = best_validation_metric.epoch
    elif metric in {
        '90% val acc', '95% val acc', '99% val acc', '99.9% val acc'
    }:
      accuracy = float(metric.split('%')[0])
      matching_rows = self.epoch_logs[(self.epoch_logs['type'] == 'val') &
                                      (self.epoch_logs['accuracy'] > accuracy)]
      epoch_num = self.epoch_logs.iloc[matching_rows.index[0]].epoch
    else:
      raise ValueError(f"Unknown metric `{metric}`")

    return self.GetEpochLogs(epoch_num)

  def ReconstructBatchFromLog(
      self, log: log_database.BatchLogMeta) -> graph_batcher.GraphBatch:
    """Reconstruct a graph batch at the given global step."""
    if not log.batch_log:
      raise OSError("Cannot re-create batch without detailed logs")

    with prof.Profile(lambda t: ('Reconstructed graph batch with '
                                 f'{graph_batch.graph_count} graphs')):
      with self.graph_db.Session() as session:
        query = session.query(graph_database.GraphMeta)
        query = query.options(sql.orm.joinedload(
            graph_database.GraphMeta.graph))
        query = query.filter(graph_database.GraphMeta.id.in_(log.graph_indices))
        # Load the graphs in the same order as in the batch.
        graphs = sorted(query.all(),
                        key=lambda g: log.graph_indices.index(g.id))

      graph_batch = graph_batcher.GraphBatch.CreateFromGraphMetas(
          graphs=(g for g in graphs),
          stats=self.batcher.stats,
          options=graph_batcher.GraphBatchOptions())

    return graph_batch

  def ReconstructBatchAtStep(self,
                             global_step: int) -> graph_batcher.GraphBatch:
    """Reconstruct a graph batch at the given global step."""
    with self.log_db.Session() as session:
      query = session.query(log_database.BatchLogMeta)
      query = query.filter(log_database.BatchLogMeta.run_id == self.run_id)
      query = query.filter(log_database.BatchLogMeta.global_step == global_step)
      query = query.limit(1)

      log: log_database.BatchLogMeta = query.first()

      if not log:
        raise OSError(f"No log found for run {self.run_id} at global step "
                      f"{global_step}")

      return self.ReconstructBatchFromLog(log)

  def GetInputOutputGraphsFromLog(self, log: log_database.BatchLogMeta):
    batch = self.ReconstructBatchFromLog(log)

    with prof.Profile(f"Recreated {batch.graph_count} input graphs"):
      input_graphs = list(batch.ToNetworkXGraphs())

    with prof.Profile(f"Recreated {batch.graph_count} output graphs"):
      # Create a duplicate graph batch which has the model predictions set
      # in place of the node_y or graph_y attributes.
      output_graph_batch = graph_batcher.GraphBatch(
          adjacency_lists=batch.adjacency_lists,
          edge_positions=batch.edge_positions,
          incoming_edge_counts=batch.incoming_edge_counts,
          node_x_indices=np.zeros(len(batch.node_x_indices)),
          node_y=log.predictions if batch.has_node_y else None,
          graph_x=None,
          graph_y=log.predictions if batch.has_graph_y else None,
          graph_nodes_list=batch.graph_nodes_list,
          graph_count=batch.graph_count,
          log=batch.log,
      )

    output_graphs = list(output_graph_batch.ToNetworkXGraphs())

    # Annotate the graphs with their GraphMeta.id column value.
    for graph_id, input_graph, output_graph in zip(log.graph_indices,
                                                   input_graphs, output_graphs):
      input_graph.id = graph_id
      output_graph.id = graph_id

    return list(zip(input_graphs, output_graphs))

  def GetInputOutputGraphsForRandomBatch(self,
                                         epoch_num: int,
                                         epoch_type: str = 'test'):
    """Reconstruct nx.MultiDiGraphs for a random batch from the given epoch_num
    where the accuracy was < 100%.

    Each graph has a 'id' property set, which is the value of GraphMeta.id
    column for the given graph.

    Returns:
      A list of <input,output> graph tuples, where each input graph is annotated
      with features and labels, and each output graph is annotated with
      predictions.
    """
    # Select a random batch with imperfect results.
    batches = self.batch_logs[(self.batch_logs['epoch'] == epoch_num) &
                              (self.batch_logs['type'] == epoch_type) &
                              (self.batch_logs['accuracy'] < 100)]

    random_row = batches.iloc[random.randint(0, len(batches) - 1)]

    input_graph_batch = self.ReconstructBatchAtStep(random_row.global_step)

    with prof.Profile(
        f"Recreated {input_graph_batch.graph_count} input graphs"):
      input_graphs = list(input_graph_batch.ToNetworkXGraphs())

    with prof.Profile(
        f"Recreated {input_graph_batch.graph_count} output graphs"):
      # Create a duplicate graph batch which has the model predictions set
      # in place of the node_y or graph_y attributes.
      output_graph_batch = graph_batcher.GraphBatch(
          adjacency_lists=input_graph_batch.adjacency_lists,
          edge_positions=input_graph_batch.edge_positions,
          incoming_edge_counts=input_graph_batch.incoming_edge_counts,
          node_x_indices=np.zeros(len(input_graph_batch.node_x_indices)),
          node_y=log.predictions if input_graph_batch.has_node_y else None,
          graph_x=None,
          graph_y=log.predictions if input_graph_batch.has_graph_y else None,
          graph_nodes_list=input_graph_batch.graph_nodes_list,
          graph_count=input_graph_batch.graph_count,
          log=input_graph_batch.log,
      )

    output_graphs = list(output_graph_batch.ToNetworkXGraphs())

    # Annotate the graphs with their GraphMeta.id column value.
    for graph_id, input_graph, output_graph in zip(log.graph_indices,
                                                   input_graphs, output_graphs):
      input_graph.id = graph_id
      output_graph.id = graph_id

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
                         output_graph: nx.MultiDiGraph,
                         averaging_method: str = 'weighted'):
  """Return the classification accuracy of the given input/output graph.

  Supports node-level or graph-level labels.

  Returns:
    Accuracy in the range 0 <= x <= 1.
  """
  try:
    true_y = np.argmax(input_graph.y)
    pred_y = np.argmax(output_graph.y)
    output_graph.correct = true_y == pred_y
    labels = np.arange(2, dtype=np.int32)
  except AttributeError:
    true_y = np.argmax([y for _, y in input_graph.nodes(data='y')], axis=1)
    pred_y = np.argmax([y for _, y in output_graph.nodes(data='y')], axis=1)
    labels = np.arange(len(input_graph.nodes[0]['y']), dtype=np.int32)
    correct = (true_y == pred_y)
    for i, (_, data) in enumerate(output_graph.nodes(data=True)):
      data['correct'] = correct[i]

  output_graph.accuracy = (true_y == pred_y).mean()
  output_graph.precision = sklearn.metrics.precision_score(
      true_y, pred_y, labels=labels, average=averaging_method)
  output_graph.recall = sklearn.metrics.recall_score(true_y,
                                                     pred_y,
                                                     labels=labels,
                                                     average=averaging_method)
  output_graph.f1 = sklearn.metrics.f1_score(true_y,
                                             pred_y,
                                             labels=labels,
                                             average=averaging_method)
  return output_graph.accuracy


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
